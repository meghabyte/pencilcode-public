from transformers import GPT2LMHeadModel 
import torch

from src.global_vars import MAX_NUM_USERS

class StudentEmbedding(torch.nn.Module):
        def __init__(self, hash_dim=MAX_NUM_USERS, embedding_dim=768, hidden_dim=64):
            super(StudentEmbedding, self).__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hash_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, embedding_dim)
            )

            # TODO: get rid of this eventually
            # initialize mlp linear layers to be zeros
            # for layer in self.mlp:
            #     if isinstance(layer, torch.nn.Linear):
            #         torch.nn.init.zeros_(layer.weight)
            #         torch.nn.init.zeros_(layer.bias)

            #         print(f'Initialized layer {layer} with zeros: {layer.weight}')

        def forward(self, hashed_user_id):
            # print number of non zeros in hashed_user_id
            # print(f'Number of non zeros in hashed_user_id: {torch.count_nonzero(hashed_user_id)}')
            # print dimension where hashed_user_id is not zero
            # print(f'Dimension where hashed_user_id is not zero: {torch.nonzero(hashed_user_id)[:, 1]}')
            # breakpoint()
            return self.mlp(hashed_user_id.float())

class StudentModel(GPT2LMHeadModel):
    def __init__(self, config, do_freeze_lm=False, do_freeze_bias=False):
        super().__init__(config)
        self.student_embedding_model = StudentEmbedding(hash_dim=MAX_NUM_USERS, embedding_dim=768, hidden_dim=64)
        print("Custom StudentModel initialized!")
        
        self.do_freeze_lm = do_freeze_lm
        self.do_freeze_bias = do_freeze_bias
        if self.do_freeze_lm:
            print(f'Freezing the LM weights')
            for param in self.transformer.parameters():
                param.requires_grad = False
                
            for param in self.lm_head.parameters():
                param.requires_grad = False
                
            # breakpoint() 
            # 
        if self.do_freeze_bias:
            print(f'Freezing the bias terms in mlp')
            self.student_embedding_model.mlp._modules['0'].bias.requires_grad = False
            self.student_embedding_model.mlp._modules['2'].bias.requires_grad = False
        
            assert self.student_embedding_model.mlp._modules['0'].bias.requires_grad == False
            assert self.student_embedding_model.mlp._modules['2'].bias.requires_grad == False
                
        # make sure number of parameters in self.transformer, self.lm_head, and self.student_embedding_model sum to the total number of parameters in the model, bc code above assumes this
        assert len(list(self.lm_head.parameters())) + len(list(self.transformer.parameters())) + len(list(self.student_embedding_model.parameters())) == len(list(self.parameters()))
           
        
    def update_past_key_values(self, past_key_values, extra_token_embedding, num_heads, head_dim):
        batch_size = extra_token_embedding.size(0)
        seq_len = 1  # The extra token length
        extra_token_kv = extra_token_embedding.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        updated_past_key_values = []
        for layer_past in past_key_values:
            past_keys, past_values = layer_past 
            updated_keys = torch.cat([extra_token_kv, past_keys], dim=2)  # Along seq_len dimension
            updated_values = torch.cat([extra_token_kv, past_values], dim=2)  # Along seq_len dimension
            updated_past_key_values.append((updated_keys, updated_values))
        return tuple(updated_past_key_values)
    
    def forward(self, input_ids, username_hashes, inputs_embeds=None, labels=None, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None: #for generating more then one token, the student embedding has already been added
            input_embeddings = self.get_input_embeddings()(input_ids)
            outputs = super().forward(inputs_embeds= input_embeddings, attention_mask=attention_mask,labels=labels, past_key_values=past_key_values, **kwargs)
            return outputs   
         
        #concatenate student embeddings
        input_embeddings = self.get_input_embeddings()(input_ids)
        one_hot_usernames = torch.nn.functional.one_hot(username_hashes, num_classes=MAX_NUM_USERS)
        student_embeddings = self.student_embedding_model(one_hot_usernames)
        new_input_embeddings = torch.cat((student_embeddings.unsqueeze(1), input_embeddings), dim=1)

        if labels is not None:
            labels = torch.cat([torch.full((labels.size(0), 1), -100, device=labels.device), labels], dim=1)
        if attention_mask is not None: 
            extra_attention = torch.ones(new_input_embeddings.size(0), 1, device=input_embeddings.device)
            attention_mask = torch.cat([extra_attention, attention_mask], dim=1)
        outputs = super().forward(inputs_embeds=new_input_embeddings, attention_mask=attention_mask,labels=labels, past_key_values=past_key_values, **kwargs)
        return outputs