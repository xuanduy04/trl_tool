import torch
from typing import Dict, Tuple, List, Union, Any

from collections.abc import Mapping


class TensorHelper:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def prepare_input(self, data: Union[torch.Tensor, Any], device) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        Copied from huggingface's `Trainer._prepare_input()`
        """
        if isinstance(data, Mapping):
            return type(data)({k: self.prepare_input(v, device) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self.prepare_input(v, device) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": device}
            return data.to(**kwargs)
        return data

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                             keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()
        
        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
    
    def move_padding_to_one_side(self, tensor: torch.Tensor, move_pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Moves all padding tokens to one side, additionally returns how to move the indicies around."""
        mask = tensor != self.pad_token_id if move_pad_to_left else tensor == self.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                                 move_pad_to_left: bool = True) -> torch.Tensor:
        """Concatenate tensors then move all paddings to one side."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.move_padding_to_one_side(concatenated, move_pad_to_left)
        return padded_tensor

    def pad_inactive_responses(self, responses_ids: torch.Tensor,
                               responses_text: List[str],
                               active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for inactive sample (in a batch) with pad tokens.
        """
        assert active_mask.sum() == responses_ids.shape[0]
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses_ids.shape[1]
        padded_responses_ids = torch.full(
            (batch_size, seq_len), self.pad_token_id,
            dtype=responses_ids.dtype, device=responses_ids.device
        )
        padded_responses_ids[active_mask] = responses_ids
        
        # Create masked response strings
        padded_responses_text = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_text[i] = responses_text[s]
                s += 1
                
        return padded_responses_ids, padded_responses_text
