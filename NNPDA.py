import tensorflow as tf
import math

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

class nnpda_cell(tf.nn.rnn_cell.RNNCell):
    """The RNN cell used in NNPDA.
        Since the pre-built cells of tensorflow like LSTM, GRU and not
        useful for NNPDA, building an nnpda_cell to call at each time step.
        
        Args:
            inputs:
                input_sym: Input symbol at time 't'.                              shape [Ni x 1]
                curr_state: Internal state at time 't' (or) current state.        shape [Ns x 1]
                curr_read: Reading of the stack at time 't' (or) current reading. shape [Nr x 1]
                st_activation: Activation function for computing internal state
                axn_activation: Activation function for computing stack action
                Ws: Weight matrix for state dynamics.                             shape [Ns x Ns x Nr x Ni]
                st_bias: Bias for computing internal state                        shape [Ns x 1]
                Wa: Weight matrix for action dynamics.                            shape [2^Ns x Nr x Ni]
                axn_bias: Scalar bias for computing the stack action
                delta: Delta matrix for computing P                                 shape [2^Ns x Ns]
                delta_: One minus delta matrix for computing P                      shape [2^Ns x Ns]
                
            outputs:
                next_state: Internal state at time 't+1' (or) next state.         shape [Ns x 1]
                next_read: Reading of the stack at time 't+1' (or) next reading.  shape [Nr x 1]
                stack_axn: Stack action at time 't+1'.                            shape [1 x 1]
    """
    
    def __init__(self, input_sym=None, curr_state=None,
                 curr_read=None, st_activation=sigmoid,
                 axn_activation=tanh, Ws, st_bias,
                 Wa, axn_bias, delta, delta_):
    
    self.input_sym = input_sym             # Input symbol received at time 't'.     shape [Ni x 1]
    self.curr_state = curr_state           # Internal state at time 't'.            shape [Ns x 1]
    self.curr_read = curr_read             # Stack reading at time 't'.             shape [Nr x 1]
    self.st_activation = st_activation     # State activation function.
    self.axn_activation = axn_activation   # Action activation function.
    self.Ws = Ws                            # Weight matrix for internal state.      shape [Ns x Ns x Nr x Ni]
    self.st_bias = st_bias                  # Bias vector for internal state.        shape [Ns x 1]
    self.Wa = Wa                            # Weight matrix for stack action.        shape [2^Ns x Nr x Ni]
    self.axn_bias = axn_bias                # Scalar bias for stack action.
    self.delta = delta                      # Delta Matrix
    self.delta_ = delta_                    # One minus delta matrix

    def __call__():
        with vs.variable_scope(scope or type(self).__name__):
            WI_s = tf.reduce_sum(tf.tensordot(self.Ws, input_sym, axes=1), axis=-1)      # The product Ws*I     shape [Ns x Ns x Nr]
            WIR_s = tf.reduce_sum(tf.tensordot(WI_s, self.curr_read, axes=1), axis=-1)   # The product Ws*I*R   shape [Ns x Ns]
            WIRS = tf.tensordot(WIR_s, self.curr_state, axes=1)                          # The product Ws*I*R*S shape [Ns x 1]
            WIRS_bias = tf.nn.bias_add(WIRS, self.st_bias)      # Adding the state bias            shape [Ns x 1]
            next_state = self.st_activation(WIRS_bias)          # Applying the activation function shape [Ns x 1]

            WI_a = tf.reduce_sum(tf.tensordot(Wa, self.input_sym, axes=1), axis=-1)      # The product Wa*I    shape [2^Ns x Nr]
            WIR_a = tf.reduce_sum(tf.tensordot(WI_a, self.curr_read, axes=1), axis=-1)   # The product Wa*I*R  shape [2^Ns]

            Sdelta = tf.multiply(self.delta, tf.transpose(tf.reverse(self.curr_state, dims=1)))         # The product delta*S          shape [2^Ns x 1]
            Sdelta_= tf.multiply(self.delta_, tf.transpose(tf.reverse(1-self.curr_state, dims=1)))      # The product (1-delta)*(1-S)  shape [2^Ns x 1]
            P = tf.reduce_prod(Sdelta + Sdelta_, axis=1)                                                # P matrix                     shape [2^Ns x 1]

            WIRP = tf.reduce_sum(tf.tensordot(WIR_a, P), axis=-1)   # Scalar stack action value
            WIRP_bias = tf.nn.bias_add(WIRP, self.axn_bias)         # Adding the scalar action bias
            stack_axn = self.axn_activation(WIRP_bias)              # Applying the activation function
            
    return next_state, stack_axn

def get_delta(K):
    """This function returns the delta matrix needed calculting Pj = delta*S + (1-delta)*(1-S)
        
        Args:
            inputs:
                K: Integers below 2^K will be considered
            outputs:
                delta: Matrix containing binary codes of numbers (1, 2^K) each one arranged row-wise.                           shape [2^K x K]
                one_minus_delta: Matrix containing complement of binary codes of numbers (1, 2^K) each one arranged row-wise.   shape [2^K x K]
    """
    delta = np.arange(1,2**K)[:,np.newaxis] >> np.arange(K)[::-1] & 1
    all_ones = np.array([list(np.binary_repr(2**int(np.ceil(np.log2(1+x)))-1, K)) for x in range(1,2**K)], dtype=int)
    one_minus_delta = all_ones - delta

    return delta, one_minus_delta

class Stack:
    def __init__(self):
        self.items = []
            
    def isEmpty(self):
        return self.items == []
                    
    def push(self, item):
        self.items.append(item)
                            
    def pop(self):
        return self.items.pop()
    
    def update(self, item):
        self.items[len(self.items)-1] = item
        return self.items
                                    
    def peek(self):
        return self.items[len(self.items)-1]
                                            
    def size(self):
        return len(self.items)

########################### Creating the Graph ###############################
def nnpda_full_3rd_order(
                         Ns, Ni, Na,
                         batch_size, str_len,
                         optimizer=rmsprop,
                         activation='sigmoid'
                         ):
    
    words = tf.placeholder(tf.int32, [batch_size, Ni , num_steps])      # Placeholder for the inputs in a given iteration.
    st_desired = tf.placeholder(tf.int32, [batch_size, Ns, num_steps])  # Placeholder for the desired final state
    
    Ws = tf.get_variable('Ws', [Ns, Ns, Nr, Ni])                # Weight matrix for computing internal state.
    bs = tf.get_variable('bs', [Ns, 1])                         # Bias vector for computing internal state.
    Wa = tf.get_variable('Wa', [2**Ns, Nr, Ni])                 # Weight matrix for computing stack action.
    ba = tf.get_variable('ba', [1, 1])                          # Scalar bias for computing stack action.
    
    cell = nnpda_cell                                           # Creating an instance for the NNPDA cell created above.
    
    initial_state = curr_state = tf.zeros([batch_size, Ns, 1])      # Initial state of the NNPDA cell
    initial_read = curr_read = tf.zeros([batch_size, Nr, 1])        # Initial reading of the stack
    delta, one_minus_delta = get_delta(Ns)                          # The delta matrices required to compute P

    sym_stack = Stack() # Stack for storing the input symbols
    len_stack = Stack() # Stack for storing the lengths of input symbols

    for i in range(num_steps):
    ############# STACK ACTION #############
        # (Default) Pushing for the initial time step
        if i == 0:
            sym_stack.push(words[:, i])
            len_stack.push(tf.norm(words[:, i], axis=-1))
        # Pushing if At > 0
        elif stack_axn > 0:
            sym_stack.push(words[:, i])
            len_stack.push(stack_axn*tf.norm(words[:, i], axis=-1))
        # Popping if At < 0
        elif stack_axn < 0:
            len_popped = 0
            # Popping a total of length |At| from the stack
            while(len_popped != -stack_axn):
                # If len(top) > |At|, Updating the length
                if len_stack.peek() > -stack_axn:
                    len_popped += -stack_axn
                    len_stack.update(len_stack.peek()-stack_axn)
                # If len(top) < |At|, Popping the top
                else
                    len_popped += len_stack.peek()
                    sym_stack.pop()
                    len_stack.pop()
        # No action if At=0
        else:
            continue
    ############# READING THE STACK ##########
        curr_read = tf.zeros([batch_size, Nr, 1])
        len_read = 0
        # Reading a total length '1' from the stack
        while(len_read != 1):
            if len_stack.peek() < 1:
                curr_read += tf.multiply(sym_stack.peek(), len_stack.peek())
                len_read += len_stack.peek()
            else:
                curr_read += sym_stack.peek()
                len_read = 1

        next_state, stack_axn = cell(words[:, i],
                                     curr_state=curr_state,
                                     curr_read=curr_read,
                                     st_activation=sigmoid,
                                     axn_activation=tanh,
                                     Ws=Ws,
                                     st_bias=st_bias,
                                     Wa=Wa,
                                     axn_bias=axn_bias,
                                     delta=delta,
                                     delta_=delta_)
        curr_state = next_state

    # Computing the Loss E = (Sf - S(t=T))^2 + (L(t=T))^2
    loss_per_example = tf.square(tf.norm(st_desired - curr_state)) + tf.square(len_stack.peek())
    total_loss = tf.reduce_mean(loss_per_example)

    return total_loss
############################## Training #####################################
#def train_nnpda():
#    with tf.Session as sess:
#        for idx, epoch in enumerate():
# This has to be written according to the data format.

