import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
from tlc_baselines.agent import RLAgent


class EncoderBlock(keras.layers.Layer):
  def __init__(self,input_size,embedding_size,hidden_layer_sizes=(128,64,32),hidden_layer_activations=(tf.nn.relu,tf.nn.relu,tf.nn.relu),output_activation=tf.nn.tanh):
      super(EncoderBlock, self).__init__()
      self.input_size= input_size
      self.embedding_size= embedding_size
      self.encoder= keras.models.Sequential();
      for i,_ in enumerate(hidden_layer_sizes):
        self.encoder.add(keras.layers.Dense(units=hidden_layer_sizes[i],activation=hidden_layer_activations[i]))
      self.encoder.add(keras.layers.Dense(units=embedding_size,activation=output_activation))
  def call(self,inputs):
    return self.encoder(inputs)

class AttentionBlock(keras.layers.Layer):
  def __init__(self):
    super(AttentionBlock, self).__init__()

  def build(self,input_shape):
    self.w = self.add_weight(shape=(input_shape[-1],input_shape[-1]),initializer="random_normal",trainable=True)
   
     
  def call(self, inputs):
      transposed_input= tf.transpose(inputs,perm=[0,2,1])
      attention_values= tf.matmul(inputs,tf.matmul(self.w, transposed_input))
      return tf.nn.softmax(tf.transpose(attention_values,perm=[0,2,1]))
  

class GraphConvLayer(keras.layers.Layer):
  def __init__(self,layer_activation=keras.activations.tanh):
    super(GraphConvLayer, self).__init__()
    self.layer_activation= layer_activation

  def build(self, input_shape):
    self.w= self.add_weight(shape=(input_shape[-1],input_shape[-1]),initializer="random_normal",trainable=True)
    self.b= self.add_weight(shape=(input_shape[-1],),initializer="zeros",trainable=True)


  def call(self, inputs, adjacency_matrix):
      temp= tf.matmul(inputs,self.w) + self.b
      temp= tf.matmul(adjacency_matrix,temp)
      return self.layer_activation(temp)


class GraphConvNetwork(keras.layers.Layer):
  def __init__(self,layer_activations=[keras.activations.tanh,keras.activations.tanh,keras.activations.tanh]):
      super(GraphConvNetwork, self).__init__()
      self.layers=[]
      for i,activation in enumerate(layer_activations):
        self.layers.append(GraphConvLayer(activation))
  def call(self, inputs, adjacency_matrix):
    output= inputs
    for layer in self.layers:
      output= layer(output,adjacency_matrix)
    return output


class DICG(keras.layers.Layer):
  def __init__(self,obervations_size,embedding_size):
    super(DICG,self).__init__()
    self.encoder= EncoderBlock(obervations_size,embedding_size)
    self.attention= AttentionBlock()
    self.GCN= GraphConvNetwork()


  def call(self, inputs):
    encoded_input= self.encoder(inputs)
    e0= encoded_input
    adj_matrix= self.attention(encoded_input)
    em= self.GCN(e0, adj_matrix)
    em= em+e0
    return em

class MLPNetwork(keras.layers.Layer):
  def __init__(self,input_size,hidden_layer_sizes,hidden_layer_activations,output_activation,output_size):
      super(MLPNetwork, self).__init__()
      self.input_size= input_size
      self.model= keras.models.Sequential();
      for i,_ in enumerate(hidden_layer_sizes):
        self.model.add(keras.layers.Dense(units=hidden_layer_sizes[i],activation=hidden_layer_activations[i]))
      self.model.add(keras.layers.Dense(units=output_size,activation=output_activation))
  def call(self, inputs):
    return self.model(inputs)

class Baseline(keras.layers.Layer):
    def __init__(self,obs_size,embedding_size,n_agents=1,hidden_layer_sizes=(512,256,128,64)
    ,hidden_layer_activations=(tf.nn.leaky_relu,tf.nn.leaky_relu,tf.nn.leaky_relu,tf.nn.leaky_relu),output_activation=None,output_size=1):
      super(Baseline, self).__init__()
      self.n_agents=n_agents
      self.embedding_size=embedding_size
      self.obs_size=obs_size
      self.DICG= DICG(obs_size,embedding_size)
      self.critic= MLPNetwork(embedding_size,hidden_layer_sizes,hidden_layer_activations,output_activation,output_size)

    def call(self, inputs,test=False):
      embeddings= self.DICG(inputs)
      return self.critic(embeddings)

    def train(self,returns,states,optimizer,mini_batch_size,epochs,verbose=True):
      batch_size= returns.shape[0]
      ind= np.random.permutation(batch_size)
      returns,states= returns[ind],states[ind]
      for epoch in range(epochs):
        batch_number=0
        baseline_losses=0
        for index in range(0,batch_size,mini_batch_size):
            batch_number+=1
            end_index=min(index+mini_batch_size,batch_size)
            baseline_loss=self.train_once(returns[index:end_index],states[index:end_index],optimizer)
            baseline_losses+=baseline_loss
        if verbose:
          print("Baseline_loss for epoch {} is {}".format(epoch,baseline_losses))
      return baseline_losses


    def train_once(self, returns, states, optimizer):
      mse= tf.keras.losses.MeanSquaredError()
      with tf.GradientTape() as baseline_tape:
        baseline= self.call(states)
        baseline_loss=mse(returns,baseline)
      grad= baseline_tape.gradient(baseline_loss,self.trainable_variables,unconnected_gradients= tf.UnconnectedGradients.NONE)
      grad, norm = tf.clip_by_global_norm(grad, 20.0)
      optimizer.apply_gradients(zip(grad,  self.trainable_variables))
      return baseline_loss

class Policy(keras.layers.Layer):
  def __init__(self,obs_size,action_space_size,hidden_layer_sizes=(512,256,128,64),hidden_layer_activations=(tf.nn.tanh,tf.nn.tanh,tf.nn.tanh,tf.nn.tanh),
               output_activation=None):
    super(Policy, self).__init__()
    self.obs_size= obs_size
    self.action_space_size= action_space_size
    self.model=  MLPNetwork(obs_size,hidden_layer_sizes,hidden_layer_activations,output_activation,action_space_size)

  def call(self, inputs):
    means= self.model(inputs)
    return means

  def act(self, inputs, greedy=False):
    means= self(inputs)
    distribution= tfp.distributions.Categorical(logits=means) 
    if greedy:
      return np.argmax(distribution.probs_parameter())
    else:
      return distribution.sample()

  def train(self,gae,states,actions,log_probs,optimizer,mini_batch_size,epochs,verbose=True,ppo_epsilon=0.2,c2=1.0):
      batch_size= states.shape[0]
      ind= np.random.permutation(batch_size)
      gae,states,actions,log_probs= gae[ind],states[ind],actions[ind],log_probs[ind]
      for epoch in range(epochs):
        batch_number=0
        policy_losses=0 
        for index in range(0,batch_size,mini_batch_size):
            batch_number+=1
            end_index=min(index+mini_batch_size,batch_size)
            policy_loss= self.train_once(gae[index:end_index],states[index:end_index],
                                         actions[index:end_index],log_probs[index:end_index],optimizer,ppo_epsilon,c2)
            policy_losses+=policy_loss
        if verbose:
          print("Policy_loss for epoch {} is {}".format(epoch,policy_loss/batch_number))
      return policy_losses

  def train_once(self, gae, states, actions, log_probs, optimizer,ppo_epsilon,c2):

    with tf.GradientTape() as policy_tape:
      means= self.call(states)
      distribution= tfp.distributions.Categorical(logits=means) 
      entropy= distribution.entropy()
      current_log_probs=distribution.log_prob(actions)
      old_log_probs= log_probs
      diff=current_log_probs-old_log_probs
      selection_ratio= tf.math.exp(diff)
      selection_ratio=tf.reshape(selection_ratio,gae.shape)
      loss=selection_ratio*gae
      clipped_loss= tf.clip_by_value(selection_ratio,1-ppo_epsilon,1+ppo_epsilon)*gae
      loss= -tf.math.minimum(loss,clipped_loss)
      loss+= -c2*entropy
      loss= tf.reduce_mean(loss)
    grad= policy_tape.gradient(loss, self.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.NONE)
    grad, norm = tf.clip_by_global_norm(grad, 20.0)
    optimizer.apply_gradients(zip(grad,  self.trainable_variables))
    return loss


class MyAgent(RLAgent):
  def __init__(self,action_space,ob_generator,reward_generator,policy):
    self.policy=policy
    self.current_phase=tf.one_hot(0,action_space.n)
    super(MyAgent, self).__init__(action_space,ob_generator,reward_generator)

  def get_action(self, ob,greedy=False):
      action= self.policy.act(ob,greedy)
      if greedy:
        self.current_phase=tf.one_hot(action,self.action_space.n)
        return action
      else:
        self.current_phase=tf.one_hot(action.numpy()[0],self.action_space.n)
        return action.numpy()[0]
  
  def get_ob(self):

      return np.append(self.ob_generator.generate(),self.current_phase)