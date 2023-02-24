import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadAttention(layers.Layer):
    def __init__(self, n_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.q_dense = layers.Dense(d_model)
        self.k_dense = layers.Dense(d_model)
        self.v_dense = layers.Dense(d_model)
        
        self.output_dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.d_model // self.n_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_dot_product = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.d_model, tf.float32))
        
        if mask is not None:
            scaled_dot_product += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_dot_product, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.output_dense(output)

class TransformerBlock(layers.Layer):
    def __init__(self, n_heads, d_model, dff, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        attn_output = self.multi_head_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
