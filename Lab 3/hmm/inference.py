import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    for i in range(0, num_time_steps):
        forward_messages[i] = rover.Distribution({})
        for zi in all_possible_hidden_states:
            sigma = 0
            position = 1
            if (observations[i] != None):
                position = observation_model(zi)[observations[i]]
            if (i == 0 and prior_distribution[zi] != 0):
                forward_messages[i][zi] = position * prior_distribution[zi]
            elif (i != 0):
                for zi1 in forward_messages[i - 1]:
                    sigma = sigma + forward_messages[i - 1][zi1] * transition_model(zi1)[zi]
            if(position * sigma != 0):
                forward_messages[i][zi] = position * sigma
        forward_messages[i].renormalize()

    # TODO: Compute the backward messages
    for i in range(0, num_time_steps):
        backward_messages[num_time_steps - 1 - i] = rover.Distribution({})
        for zi in all_possible_hidden_states:
            sigma = 0
            if (i == 0):
                backward_messages[num_time_steps - 1][zi] = 1
            else:
                for zi1 in backward_messages[num_time_steps - i]:
                    position = 1
                    if (observations[num_time_steps - i] != None):
                        position = observation_model(zi1)[observations[num_time_steps - i]]
                    sigma = sigma + backward_messages[num_time_steps - i][zi1] * position * transition_model(zi)[zi1]
                    if(sigma != 0):
                        backward_messages[num_time_steps - 1 - i][zi] = sigma
        backward_messages[num_time_steps - 1 - i].renormalize()
    
    # TODO: Compute the marginals 
    for i in range(0, num_time_steps):
        marginals[i] = rover.Distribution({})
        sigma = 0
        for zi in all_possible_hidden_states:
            marginals[i][zi] = forward_messages[i][zi] * backward_messages[i][zi]
            sigma = sigma + forward_messages[i][zi] * backward_messages[i][zi]
        for zi in marginals[i].keys():
            marginals[i][zi] = marginals[i][zi] / sigma
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    w, z, estimated_hidden_states = ([None] * (num_time_steps) for idx in range(3))

    for i in range(0, num_time_steps):
        w[i] = rover.Distribution({})
        z[i] = dict()
        for zi in all_possible_hidden_states:
            position = 1
            if(observations[i] != None):
                position = observation_model(zi)[observations[i]]
            if(i == 0 and prior_distribution[zi] != 0):
                w[i][zi] = np.log(position * prior_distribution[zi])
            elif (i != 0):
                max = -np.inf
                for zi1 in w[i-1]:
                    if (transition_model(zi1)[zi] != 0):
                        max_term = np.log(transition_model(zi1)[zi]) + w[i-1][zi1]
                        if ( max_term > max and position != 0):
                            max = max_term
                            z[i][zi] = zi1
                if (position != 0):
                    w[i][zi] = np.log(position) + max

    max = -np.inf
    for zi in w[num_time_steps - 1]:
        max_term = w[num_time_steps - 1][zi]
        if (max_term > max):
            max = max_term
            estimated_hidden_states[num_time_steps - 1] = zi

    for i in range(1, num_time_steps):
        estimated_hidden_states[num_time_steps - 1 - i] = z[num_time_steps - i][estimated_hidden_states[num_time_steps - i]]

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    if(missing_observations):
        timestep = 30    
    else:
        timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    #errors for viterbi and forward-backward
    viterbi = forward_backward = 0
    for i in range(0, num_time_steps):
        if(hidden_states[i] == estimated_states[i]):
            viterbi = viterbi + 1
        max = 0
        index = 0
        for zi in marginals[i]:
            if (marginals[i][zi] > max):
                max = marginals[i][zi]
                index = zi
        print("state: ", i, index)
        if(hidden_states[i] == index):
            forward_backward = forward_backward + 1

    print("viterbi: ", 1-viterbi/100)
    print("forward-backward: ", 1-forward_backward/100)
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
