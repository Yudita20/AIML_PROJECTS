import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len):
        
        super().__init__()
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
        self.sos_token = torch.Tensor([self.src_tokenizer.token_to_id['[SOS]']], dtype=torch.int64)
        self.eos_token = torch.Tensor([self.src_tokenizer.token_to_id['[EOS]']], dtype=torch.int64)
        self.pad_token = torch.Tensor([self.src_tokenizer.token_to_id['[PAD]']], dtype=torch.int64)


    def __len__(self):

        return len(self.ds)


    def __getitem__(self, idx):

       scr_target_pair = self.ds[idx]
       scr_text = scr_target_pair['translation'][self.src_lang]
       tgt_text = scr_target_pair['translation'][self.tgt_lang]
      
       enc_input_tokens = self.tokenizer_src.encode(src_text).ids
       dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

       enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for [SOS] and [EOS]
       dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for [SOS]

       if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        raise ValueError(f"Sequence length {self.seq_len} is too short for the input sequences.")
       
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        decoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        label = torch.cat(
            [
                
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len


        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label" : label,
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }



def causal_mask(size):

    mask = torch.tril(torch.ones(1, size, size)).type(torch.int)
    return mask == 0
