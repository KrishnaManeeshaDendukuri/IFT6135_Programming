import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        print("######## shapes ############")
        print(self.hidden_size)
        print(inputs.shape)
        
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        var = torch.var(inputs, dim=-1, keepdim = True, unbiased=False)
        # var = torch.mean((inputs-mean)**2, -1, keepdim = True)
        scale = (inputs - mean) / torch.sqrt(var + self.eps)
        ln_output = scale * self.weight + self.bias
        return ln_output

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        # self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        # self.register_parameter('q_proj_weight', None)
        # self.register_parameter('k_proj_weight', None)
        # self.register_parameter('v_proj_weight', None)

        # self.register_parameter('in_proj_bias', None)
        # self.bias_q = self.bias_k = self.bias_v = None
        # self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias)

        # self.q_proj_weight = nn.Parameter(torch.Tensor(self.head_size))
        # self.k_proj_weight = nn.Parameter(torch.Tensor(self.head_size))
        # self.v_proj_weight = nn.Parameter(torch.Tensor(self.head_size))

        # self.bias_q = nn.Parameter(torch.Tensor(self.head_size))
        # self.bias_k = nn.Parameter(torch.Tensor(self.head_size))
        # self.bias_v = nn.Parameter(torch.Tensor(self.head_size))

        # self.out_proj_weight = nn.Parameter(torch.Tensor(self.head_size))
        # self.bias_out = nn.Parameter(torch.Tensor(self.head_size))
        d_model = self.head_size* self.num_heads
        self.q_linear = nn.Linear(d_model, d_model, bias=True)
        self.k_linear = nn.Linear(d_model, d_model, bias=True)
        self.v_linear = nn.Linear(d_model, d_model, bias=True)
        self.final_linear = nn.Linear(d_model, d_model, bias=True)

        # ==========================
        # TODO: Write your code here
        # ==========================

    def get_attention_weights(self, queries, keys):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        scores = torch.matmul(queries, keys.transpose(-2, -1)) /  math.sqrt(self.head_size)
        scores = F.softmax(scores, dim=-1)
        return scores

    def apply_attention(self, queries, keys, values):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. 

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. 
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. 
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        print("apply_attention impln")
        attention_weights = self.get_attention_weights(queries, keys)
        attention_heads = torch.matmul(attention_weights, values)
        print(attention_heads.shape)
        merged_heads = self.merge_heads(attention_heads)
        print(merged_heads.shape)
        return merged_heads

    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        print("split_heads impln")
        batch_size, seq_len, merged_dim = tensor.shape
        dim = merged_dim // self.num_heads
        tensor = tensor.view(batch_size, seq_len, self.num_heads, dim)
        output = tensor.transpose(1,2)
        print(output.shape)
        return output

    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        print("merge_heads impln")
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.transpose(1, 2)
        output = tensor.contiguous().view(batch_size, seq_len, -1)
        print(output.shape)
        return output

    def forward(self, hidden_states):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        print(hidden_states.shape)
        # multi_heads = self.split_heads(hidden_states)
        # print(hidden_states.shape)
        # print(multi_heads.shape)
        # Q = torch.matmul(multi_heads, self.q_proj_weight) +self.bias_q
        # K = torch.matmul(multi_heads, self.k_proj_weight) +self.bias_k
        # V = torch.matmul(multi_heads, self.v_proj_weight) +self.bias_v
        print("forward impln")
        Q = self.q_linear(hidden_states)
        K = self.k_linear(hidden_states)
        V = self.v_linear(hidden_states)
        
        print(Q.shape, K.shape, V.shape)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        Y = self.apply_attention(Q,K,V)
        print(Y.shape)
        output = self.final_linear(Y)
        return output



class PostNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads,sequence_length, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
       
        attention_outputs = self.attn(x)
        #print(inp_x.shape)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs=self.linear(attention_outputs)

        outputs = self.layer_norm_2(outputs+attention_outputs)
        return outputs

class PreNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads,sequence_length, dropout=0.0):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.

        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            sequence_length - Length of the sequence
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        # ==========================
        # TODO: Write your code here
        # ==========================
        layernorm_1_out = self.layer_norm_1(x)
        attention_out = self.attn(layernorm_1_out)
        add_out = x + attention_out

        layernorm_2_out = self.layer_norm_2(add_out)
        ffn_out = self.linear(layernorm_2_out)   

        output = add_out + ffn_out
        return output


class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim=256, hidden_dim=512, num_channels=3, num_heads=8, num_layers=4, num_classes=10, patch_size=4, num_patches=64,block='prenorm', dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            block - Type of attention block
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
            
        """
        super().__init__()
        
        self.patch_size = patch_size
        #Adding the cls token to the sequnence 
        self.sequence_length= 1+ num_patches
        # Layers/Networks
        print(dropout)
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        if block =='prenorm':
          self.transformer = nn.Sequential(*[PreNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        else:
          self.transformer = nn.Sequential(*[PostNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,self.sequence_length,embed_dim))
    
    def get_patches(self,image, patch_size, flatten_channels=True):
        """
        Inputs:
            image - torch.Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                              as a feature vector instead of a image grid.
        Output : torch.Tensor representing the sequence of shape [B,patches,C*patch_size*patch_size] for flattened.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        print("get patches impln")

        # embed_dim=256
        # B,C,H,W = image.shape
        # self.n_patches = (H//patch_size)*(W//patch_size)
        # print(B,C,H,W)

        # self.proj = nn.Conv2d(
        #         C,
        #         patch_size*patch_size*C,
        #         kernel_size=patch_size,
        #         stride=patch_size,
        # )
        
        # x = self.proj(image)
        # print(image.shape, x.shape)
        # if flatten_channels:
        #     image_seq = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        #     print(image_seq.shape)
        # else:
        #     image_seq = x
        # print(f"n patches = {self.n_patches}")
        # image_seq = image_seq.transpose(1, 2) 

        B, C, H, W = image.shape
        image = image.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        image = image.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
        image = image.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
        if flatten_channels:
            image = image.flatten(2,4)
        return image


    def forward(self, x):
        """ViT

        This is a small version of Vision Transformer

        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, channels,height , width)`)
            The input tensor containing the iamges.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_classes)`)
            A tensor containing the output from the mlp_head.
        """
        # Preprocess input
        x = self.get_patches(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
        
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]
        
        #Add dropout and then the transformer
        
        # ==========================
        # TODO: Write your code here
        # ==========================
        
        x = self.dropout(x)

        # x = x.transpose(0, 1)
        x = self.transformer(x)
        #Take the cls token representation and send it to mlp_head

        # ==========================
        # TODO: Write your code here
        # ==========================
        # print(f"cls_token representatiom? {x.shape}")
        # print(x[-1].shape)
        cls = x[:, 0, :]
        output = self.mlp_head(cls)
        return output
        
    def loss(self,preds,labels):
        '''Loss function.

        This function computes the loss 
        parameters:
            preds - predictions from the model
            labels- True labels of the dataset

        '''
        # ==========================
        # TODO: Write your code here
        # ==========================
        loss = F.cross_entropy(preds, labels)
        return loss        
