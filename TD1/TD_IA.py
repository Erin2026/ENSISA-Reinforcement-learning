import numpy as np

int_to_char = {
    0 : 'u',
    1 : 'r',
    2 : 'd',
    3 : 'l'
}

policy_one_step_look_ahead = {
    0 : [-1,0],
    1 : [0,1],
    2 : [1,0],
    3 : [0,-1]
}

def policy_int_to_char(pi,n):

    pi_char = ['']

    for i in range(n):
        for j in range(n):

            if i == 0 and j == 0 or i == n-1 and j == n-1:

                continue

            pi_char.append(int_to_char[pi[i,j]])

    pi_char.append('')

    return np.asarray(pi_char).reshape(n,n)

def policy_evaluation(n,pi,v,Gamma,threshold):
  """
    This function should return the value function that follows the policy pi.
    Use the stopping criteria given in the problem statement.
  """
  while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                    continue
                action = pi[i, j]
                direction = policy_one_step_look_ahead[action]
                ni, nj = i + direction[0], j + direction[1]
                if 0 <= ni < n and 0 <= nj < n:
                    v_new = -1 + Gamma * v[ni, nj]
                    delta = max(delta, abs(v[i, j] - v_new))
                    v[i, j] = v_new
        if delta < threshold:
            break
  return v

def policy_improvement(n,pi,v,Gamma):
  """
    This function should return the new policy by acting in a greedy manner.
    The function should return as well a flag indicating if the output policy
    is the same as the input policy.

    Example:
      return new_pi, True if new_pi = pi for all states
      else return new_pi, False
  """
  policy_stable = False
  for i in range(n):
      for j in range(n):
          if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
              continue
          old_action = pi[i, j]
          action_values = []
          for action, direction in policy_one_step_look_ahead.items():
              ni, nj = i + direction[0], j + direction[1]
              if 0 <= ni < n and 0 <= nj < n:
                  action_values.append((-1 + Gamma * v[ni, nj], action))
          best_action = max(action_values)
          pi[i, j] = best_action
          if best_action == old_action:
              policy_stable = True
  return pi, policy_stable
  

def policy_initialization(n):
  """
    This function should return the initial random policy for all states.
  """
  return np.random.choice([0, 1, 2, 3], size=(n, n))

def policy_iteration(n,Gamma,threshold):

    pi = policy_initialization(n=n)

    v = np.zeros(shape=(n,n))

    while True:

        v = policy_evaluation(n=n,v=v,pi=pi,threshold=threshold,Gamma=Gamma)

        pi , pi_stable = policy_improvement(n=n,pi=pi,v=v,Gamma=Gamma)

        if pi_stable:

            break

    return pi , v

n = 4

Gamma = [0.8,0.9,1]

threshold = 1e-4

for _gamma in Gamma:

    pi , v = policy_iteration(n=n,Gamma=_gamma,threshold=threshold)

    pi_char = policy_int_to_char(n=n,pi=pi)

    print()
    print("Gamma = ",_gamma)

    print()

    print(pi_char)

    print()
    print()

    print(v)