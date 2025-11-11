import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from transformers import AutoTokenizer

from .tensor_helper import TensorHelper


@dataclass
class LMGenerationConfig:
    max_turns: int = field(
        default=3,
        metadata={
            "help": "Maximum number of tool calls by the model."
        },
    )
    max_obs_length: int = field(
        default=500,
        metadata={
            "help": "Maximum length of the tool call's output (in tokens)."
        },
    )
    tool_call_tag: str = field(
        default="calculator",
        metadata={
            "help": "the HTML tag for when the LLM want to call the tool. Example: '<calculator>69+420</calculator>'."
        }
    )
    tool_output_tag: str = field(
        default="tool_output",
        metadata={
            "help": "the HTML tag containing the tool's output. Example: '<tool_output>177013</tool_output>'."
        }
    )
    invalid_observation_text: str = field(
        default=f'\nMy previous action is invalid. \
If I want to use the calculator, I should put the query between <calculator> and </calculator>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n',
        metadata={
            "help": "the text that the environment returns if the agent's action is invalid."
        }
    )

    def __post_init__(self):
        # Check for inconsistencies & argument errors here.
        if " " in self.tool_call_tag:
            raise ValueError("tool_call_tag must not contain spaces.")

        if " " in self.tool_output_tag:
            raise ValueError("tool_output_tag must not contain spaces.")

        if self.tool_call_tag == self.tool_output_tag:
            raise ValueError("tool_call_tag and tool_output_tag must be unique.")


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
        # False: (prompt) -> llm -> tool (if called) ...

        self.tensor_fn = TensorHelper(pad_token_id=self.tokenizer.pad_token_id)

    def generate(self, unwrapped_model, generate_inputs: Dict[str, Tensor], generation_config,
                 disable_compile: bool, device):
        print("BEGIN LMGenerationManager's `generate`")
        # print(f"{type(unwrapped_model)=}\n{generate_inputs=}")
        # Pre-loop:

        # left_side = {'input_ids': generate_inputs['input_ids']}
        right_side: Dict[str, Tensor] = {'responses_ids': generate_inputs['input_ids'][:, []],
                                         'responses_ids_masked_tool_output': generate_inputs['input_ids'][:, []]}

        active_mask: Tensor = torch.ones(generate_inputs['input_ids'].shape[0], dtype=torch.bool)
        turns_stats: Tensor = torch.ones(generate_inputs['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats: Tensor = torch.zeros(generate_inputs['input_ids'].shape[0], dtype=torch.int)
        valid_tool_call_stats: Tensor = torch.zeros(generate_inputs['input_ids'].shape[0], dtype=torch.int)
        active_num_list: List[int] = [active_mask.sum().item()]
        rollings: Dict[str, Tensor] = generate_inputs

        # Main loop:
        if self.tool_first:
            # TODO: detokenize generate_inputs -> call tool -> tokenize generate_inputs & update attention mask
            raise NotImplementedError

        for step in range(self.args.max_turns):
            # print(f"------ Begin {step=} ------")
            if not active_mask.sum():  # If there are no active generations
                break
            # Pre-inference
            rollings = self.tensor_fn.prepare_input(rollings, device)
            # remove padding?
            print("-------- Removing padding... ", end='')
            rollings = self.tensor_fn.cut_to_effective_len(
                rollings,
                keys=['input_ids', 'attention_mask']
            )
            print("Done")
            rollings_active = {
                k: v[active_mask] for k, v in rollings.items()
            }

            # Main inference
            print("-------- Generating responses... ", end='')
            responses_ids = unwrapped_model.generate(
                **rollings_active, generation_config=generation_config, disable_compile=disable_compile
            )
            print("Done")
            # print(f"{responses_ids.shape=}")

            # Post-inference
            print("-------- postprocess responses... ", end='')
            responses_ids, responses_text = self._postprocess_responses(responses_ids, device)
            print("Done")
            print("-------- pad_inactive_responses... ", end='')
            responses_ids, responses_text = self.tensor_fn.pad_inactive_responses(
                responses_ids, responses_text, active_mask
            )
            print("Done")

            # Execute in environment and process observations
            print("-------- execute_predictions... ", end='')
            next_obs_text, dones, is_valid_action, is_tool_call = self.execute_predictions(
                responses_text, active_mask
            )
            print("Done")

            # Update "dones" (i.e. active mask)
            curr_active_mask = ~torch.tensor(dones, dtype=torch.bool)
            active_mask &= curr_active_mask
            # Explanation:
            #   curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            #   active_mask = active_mask * curr_active_mask

            # INFOs
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(is_valid_action, dtype=torch.int)
            valid_tool_call_stats += torch.tensor(is_tool_call, dtype=torch.int)

            # Update obs
            next_obs_ids = self._process_next_obs(next_obs_text, device=device)
            # Update states
            print("-------- Update states... ", end='')
            rollings = self._update_rollings(
                rollings,
                responses_ids,
                next_obs_ids
            )
            right_side = self._update_right_side(
                right_side,
                responses_ids,
                next_obs_ids
            )
            print("Done")
            # print(f"{rollings['input_ids'].shape=}, {right_side['responses_ids'].shape=})")

        # final LLM rollout
        if active_mask.sum():
            # print(f"------ Begin final LLM rollout ------")
            rollings = self.tensor_fn.prepare_input(rollings, device)
            # rollings = self.tensor_fn.cut_to_effective_len(
            #     rollings,
            #     keys=['input_ids', 'attention_mask']
            # )
            rollings_active = {
                k: v[active_mask] for k, v in rollings.items()
            }

            # Main inference
            responses_ids = unwrapped_model.generate(
                **rollings_active, generation_config=generation_config, disable_compile=disable_compile
            )
            # print(f"{responses_ids.shape=}")

            # Post-inference
            responses_ids, responses_text = self._postprocess_responses(responses_ids, device)
            responses_ids, responses_text = self.tensor_fn.pad_inactive_responses(
                responses_ids, responses_text, active_mask
            )

            # Execute in environment and process observations
            _, dones, is_valid_action, is_tool_call = self.execute_predictions(
                responses_text, active_mask
            )

            # Update "dones" (i.e. active mask)
            curr_active_mask = ~torch.tensor(dones, dtype=torch.bool)
            active_mask &= curr_active_mask
            # Explanation:
            #   curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            #   active_mask = active_mask * curr_active_mask

            # INFOs
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(is_valid_action, dtype=torch.int)
            valid_tool_call_stats += torch.tensor(is_tool_call, dtype=torch.int)

            # Update states
            right_side = self._update_right_side(
                right_side,
                responses_ids,
            )
            print("Done")
            # print(f"{right_side['responses_ids'].shape=}")

        info = {
            'turns_stats': turns_stats.tolist(),
            'active_mask': active_mask.tolist(),
            'valid_action_stats': valid_action_stats.tolist(),
            'valid_tool_call_stats': valid_tool_call_stats.tolist(),
        }
        # print(f"{info=}")
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # final_output = self._compose_final_output(left_side, right_side, info)
        responses_ids = right_side['responses_ids']
        all_tool_output_mask = self.tensor_fn.create_attention_mask(right_side['responses_ids_masked_tool_output'])

        print("END LMGenerationManager's `generate`")
        return responses_ids, self.tensor_fn.create_attention_mask(responses_ids), all_tool_output_mask

    def _batch_tokenize(self, text: List[str], device) -> torch.Tensor:
        """Tokenize a batch of text."""
        return self.tokenizer(
            text=text,
            return_tensors="pt",
            padding="longest",
            padding_side="right",
            add_special_tokens=False,
        )['input_ids'].to(device)

    def _postprocess_responses(self, responses_ids: torch.Tensor, device) -> Tuple[Tensor, List[str]]:
        """Process responses to stop at tool call operation or answer operation."""
        responses_text: List[str] = self.tokenizer.batch_decode(
            responses_ids,
            skip_special_tokens=True
        )

        responses_text = [
            resp.split(f'</{self.args.tool_call_tag}>')[0] + f'</{self.args.tool_call_tag}>'
            if f'</{self.args.tool_call_tag}>' in resp
            else resp.split('</answer>')[0] + '</answer>' if '</answer>' in resp
            else resp
            for resp in responses_text]

        responses_ids = self._batch_tokenize(responses_text, device=device)
        return responses_ids, responses_text

    def _process_next_obs(self, next_obs_text: List[str], device) -> torch.Tensor:
        """
        Process next observations from environment.
        Tokenize observations, then truncate observations if necessary.
        """
        next_obs_ids = self._batch_tokenize(next_obs_text, device=device)

        if next_obs_ids.shape[1] > self.args.max_obs_length:
            print(
                f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]=} & {self.args.max_obs_length=}")
            next_obs_ids = next_obs_ids[:, :self.args.max_obs_length]

        return next_obs_ids

    def _update_rollings(self, rollings, responses_ids: torch.Tensor,
                         next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings['input_ids'],
            responses_ids,
            next_obs_ids
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        # new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = int(new_attention_mask.sum(dim=1).max())
        max_len = effective_len  # min(self.args.max_prompt_length, effective_len)

        new_rollings = {
            'input_ids': new_input_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
            # 'position_ids': new_position_ids[:, -max_len:],
        }
        return new_rollings

    def _update_right_side(self, right_side: Dict, curr_responses_ids: torch.Tensor,
                           next_obs_ids: Optional[torch.Tensor] = None) -> Dict:
        """Update right side state."""
        responses_ids = [right_side['responses_ids'], curr_responses_ids]
        responses_ids_masked_tool_output = [right_side['responses_ids_masked_tool_output'], curr_responses_ids]
        # responses_ids_masked_tool_output is responses_ids 
        #   but with all the tool's output turned into pad_tokens.

        if next_obs_ids is not None:
            responses_ids.append(next_obs_ids)
            next_obs_mask = torch.full(next_obs_ids.size(), self.tokenizer.pad_token_id, dtype=next_obs_ids.dtype,
                                       device=next_obs_ids.device)  # tool's output mask
            responses_ids_masked_tool_output.append(next_obs_mask)

        responses_ids = torch.cat(responses_ids, dim=1)
        responses_ids_masked_tool_output = torch.cat(responses_ids_masked_tool_output, dim=1)

        responses_ids, sorted_indices = self.tensor_fn.move_padding_to_one_side(responses_ids, move_pad_to_left=False)
        responses_ids_masked_tool_output = responses_ids_masked_tool_output.gather(1, sorted_indices)

        effective_len = int(self.tensor_fn.create_attention_mask(responses_ids).sum(dim=1).max())
        max_len = effective_len  # min(self.args.max_prompt_length, effective_len)

        return {'responses_ids': responses_ids[:, :max_len],
                'responses_ids_masked_tool_output': responses_ids_masked_tool_output[:, :max_len]}

    def _compose_final_output(self,
                              left_side: Dict,
                              right_side: Dict,
                              info: Dict) -> Dict:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts_ids'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses_ids']
        ], dim=1)

        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_ids'])
        ], dim=1)
        final_output['responses_ids_masked_tool_output'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_ids_masked_tool_output'])
        ], dim=1)

        # final_output['position_ids'] = self.tensor_fn.create_position_ids(
        #     final_output['attention_mask']
        # )
        final_output.update(info)

        return final_output

    def execute_predictions(self, predictions: List[str], active_mask=None, use_tool: bool = True) \
            -> Tuple[List[str], List[bool], List[bool], List[bool]]:
        """
        Execute model predictions across multiple environments.

        This function acts as the environment's `step` method. It processes
        the model's textual predictions into actions and contents, executes
        tool calls (e.g., calculator invocations) when required, and returns
        the corresponding next observations.

        Args:
            predictions (List[str]): Raw action predictions from the model.
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

        tool_input = [content for action, content in zip(cur_actions, contents) if action == 'tool_call']
        if use_tool:
            tool_output = self.batch_tool_call(tool_input)
            assert (len(tool_output) == cur_actions.count('tool_call'))
            # or go with sum([1 for action in cur_actions if action == 'tool_call']))
            # if you wanna be verbose about it
        else:
            tool_output = [''] * cur_actions.count('tool_call')

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
                elif action == 'tool_call':
                    next_obs.append(
                        f'\n\n<{self.args.tool_output_tag}>{tool_output.pop(0).strip()}</{self.args.tool_output_tag}>\n\n')
                    dones.append(False)
                    is_valid_action.append(True)
                    is_tool_call.append(True)
                else:
                    # Invalid actions
                    next_obs.append(self.args.invalid_observation_text)
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
                pattern = rf'<({self.args.tool_call_tag}|answer)>(.*?)</\1>'
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
        Args:
            tool_inputs: input for the tool
        Returns:
            RAW tool call outputs
        """
        return [self.tool(tool_input) for tool_input in tool_inputs]

    def _tooloutput2string(self, tool_output: Any) -> str:
        """Post process a single tool call's output."""
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
