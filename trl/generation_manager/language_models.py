import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable

import torch
from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoTokenizer


@dataclass
class LMGenerationConfig:
    max_turns: int  # maximum total number of actions BY THE LLM.
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False


class LMGenerationManager:
    """Language-model's Generation Manager"""
    def __init__(
            self,
            args: LMGenerationConfig,
            processing_class: 'AutoTokenizer',
            tool: Callable[..., str],
            tool_first: bool = False,
    ):
        self.args = args

        self.tokenizer = processing_class
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.tool = tool
        self.tool_first = tool_first
        # True:  (prompt -> tool) -> llm -> tool (if called) ...
        # False: prompt -> llm -> tool (if called) ...

    def generate(self, unwrapped_model, generate_inputs, generation_config, disable_compile : bool = True):
        # Pre-loop:
        print(f"{type(unwrapped_model)=}\n{generate_inputs=}")
        # active_mask = torch.ones(generate_inputs['input_ids'].shape[0], dtype=torch.bool)

        # Main loop:
        if self.tool_first:
            # TODO:
            raise NotImplementedError

        for step in range(self.args.max_turns):
            if not active_mask.sum():  # If there are no active generations
                break
            # Pre-inference
            pass

            # Main inference
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=generation_config, disable_compile=disable_compile
            )
            print(f"{prompt_completion_ids=}")

            # Execute in environment and process observations
            next_obs, dones, is_valid_action, is_tool_call = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            # Post-inference
            pass

        # Post-loop:

        return prompt_completion_ids, None
    
    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        original_left_side = {'input_ids': initial_input_ids[:, -self.args.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []],
                               'responses_with_tool_output_mask': initial_input_ids[:, []]}

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_tool_call_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list: List[int] = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.args.max_turns):
            if not active_mask.sum():  # If there are no active generations
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict(tensors={
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn.example_level_pad(responses_ids, responses_str,
                                                                            active_mask)

            # Execute in environment and process observations
            next_obs, dones, is_valid_action, is_tool_call = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )

            # Update "dones" (i.e. active mask)
            curr_active_mask = ~torch.tensor(dones, dtype=torch.bool)
            active_mask &= curr_active_mask
            # Explanation:
            # curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            # active_mask = active_mask * curr_active_mask

            # META INFOs
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(is_valid_action, dtype=torch.int)
            valid_tool_call_stats += torch.tensor(is_tool_call, dtype=torch.int)

            # Update obs
            next_obs_ids = self._process_next_obs(next_obs)
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )

        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn.example_level_pad(responses_ids, responses_str,
                                                                            active_mask)

            # Execute in environment and process observations
            _, dones, is_valid_action, is_tool_call = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, use_tool=False
            )
            # Update "dones" (i.e. active mask)
            curr_active_mask = ~torch.tensor(dones, dtype=torch.bool)
            active_mask &= curr_active_mask

            # META INFOs
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(is_valid_action, dtype=torch.int)
            valid_tool_call_stats += torch.tensor(is_tool_call, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_tool_call_stats'] = valid_tool_call_stats.tolist()

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[Tensor, List[str]]:
        """Process responses to stop at tool call operation or answer operation."""
        responses_str: List[str] = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )

        responses_str = [
            resp.split('</calculator>')[0] + '</calculator>' if '</calculator>' in resp
            else resp.split('</answer>')[0] + '</answer>' if '</answer>' in resp
            else resp
            for resp in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """
        Process next observations from environment.
        Tokenize, then truncate observations if necessary.
        """
        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.args.max_obs_length:
            print(
                f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.args.max_obs_length}")
            next_obs_ids = next_obs_ids[:, :self.args.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor,
                              next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = int(new_attention_mask.sum(dim=1).max())
        max_len = min(self.args.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _tool_output_masked_concatenate_with_padding(
            self,
            prompt: torch.Tensor,
            prompt_with_mask: torch.Tensor,
            response: torch.Tensor,
            info: torch.Tensor = None,
            pad_to_left: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Concatenate tensors and handle padding. Additionally, create a mask (tool_output_mask) to cover the tool's output block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            tool_output_mask = torch.full(info.size(), pad_id, dtype=info.dtype,
                                          device=info.device)  # tool's output mask
            tensors_with_mask.append(tool_output_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict,
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses, responses_with_tool_output_mask = self._tool_output_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_tool_output_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False
            )
        else:
            responses, responses_with_tool_output_mask = self._tool_output_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_tool_output_mask'],
                cur_responses,
                pad_to_left=False
            )
        effective_len = int(self.tensor_fn.create_attention_mask(responses).sum(dim=1).max())
        max_len = min(self.args.max_prompt_length, effective_len)

        return {'responses': responses[:, :max_len],
                'responses_with_tool_output_mask': responses_with_tool_output_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        return active_batch

    def _compose_final_output(self, left_side: Dict,
                              right_side: Dict,
                              meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['tool_output_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_tool_output_mask'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, use_tool: bool = True) \
            -> Tuple[List[str], List[bool], List[bool], List[bool]]:
        """
        Execute model predictions across multiple environments.

        This function acts as the environment's `step` method. It processes
        the model's textual predictions into actions and contents, executes
        tool calls (e.g., calculator invocations) when required, and returns
        the corresponding next observations.

        Args:
            predictions (List[str]): Raw action predictions from the model.
            pad_token (str): Token used for padding responses.
            active_mask (optional): Mask indicating which environments are active.
            use_tool (bool): If False, detected tool call actions are skipped
                and no tool is executed.
        Returns:
            Tuple of (next_obs, dones, is_valid_action, is_tool_call)
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs: List[str] = []
        dones: List[bool] = []
        is_valid_action: List[bool] = []
        is_tool_call: List[bool] = []

        tool_input = [content for action, content in zip(cur_actions, contents) if action == 'calculator']
        if use_tool:
            tool_output = self.batch_tool_call(tool_input)
            assert (len(tool_output) == cur_actions.count('calculator'))
            # or go with sum([1 for action in cur_actions if action == 'calculator']))
            # if you wanna be verbose about it
        else:
            tool_output = [''] * cur_actions.count('calculator')

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(True)
                is_valid_action.append(False)
                is_tool_call.append(False)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(True)
                    is_valid_action.append(True)
                    is_tool_call.append(False)
                elif action == 'calculator':
                    next_obs.append(f'\n\n<tool_output>{tool_output.pop(0).strip()}</tool_output>\n\n')
                    dones.append(False)
                    is_valid_action.append(True)
                    is_tool_call.append(True)
                else:
                    # Invalid actions
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to use the calculator, I should put the query between <calculator> and </calculator>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(False)
                    is_valid_action.append(False)
                    is_tool_call.append(False)

        assert len(tool_output) == 0, "Not all tool calls were processed. Something has gone horribly wrong."

        return next_obs, dones, is_valid_action, is_tool_call

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.

        Args:
            predictions: List of raw predictions

        Returns:
            Tuple of (list of actions, list of the action's contents)
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):  # for llm output
                pattern = r'<(calculator|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)  # tool call or answer
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)

        return actions, contents

    def batch_tool_call(self, tool_inputs: List[str] = None) -> List[str]:
        """
        Batchified tool_call for tool_input. Includes post-processing.
        Args:
            tool_inputs: input for the tool
        Returns:
            tool call outputs, converted into strings
        """
        results = self._batch_tool_call(tool_inputs)

        return [self._tooloutput2string(result) for result in results]

    def _batch_tool_call(self, tool_inputs: List[str] = None) -> List[Any]:
        """
        Batchified tool_call for tool_input. Returns RAW tool call outputs.
        (Customize for each tool)
        Args:
            tool_inputs: input for the tool
        Returns:
            RAW tool call outputs
        """
        return [(calculator(tool_input)) for tool_input in tool_inputs]

    def _tooloutput2string(self, tool_output: Any) -> str:
        """Post process a single tool call's output.
        (Customize for each tool)"""
        # # Example for search engine
        # format_reference = ''
        # for idx, doc_item in enumerate(tool_output):
        #     content = doc_item['document']['contents']
        #     title = content.split("\n")[0]
        #     text = "\n".join(content.split("\n")[1:])
        #     format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        #
        # return format_reference
        return str(tool_output)
