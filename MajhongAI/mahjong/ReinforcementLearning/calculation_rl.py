# -*- coding: utf-8 -*-
# @FileName : calculation_rl
# @Project  : MAHJONG AI
# @Author   : gangyinglau
# @Time     : 25/4/2021 3:06 AM

# Corresponding to epsilon_3 (methods.py)
def cal_probability_of_action(is_trigger_by_rl, epsilon, discard_argmax, discard_probabilities):
    p_action = 0
    try:
        p_action = (1 - epsilon) * discard_probabilities.T[discard_argmax][0]
    except Exception:
        print('Error here, type 1: experience/cal_probability_of_action')
    if not is_trigger_by_rl:
        try:
            p_action += epsilon / 27
        except Exception:
            print('Error here, type 4: experience/cal_probability_of_action')
    return p_action


# Just retrieve the discard card probability of the model
def cal_probability_of_action_2(discard_argmax, discard_probabilities):
    p_action = 0
    try:
        p_action = discard_probabilities.T[discard_argmax][0]
    except Exception:
        print('Error here, type 1: experience/cal_probability_of_action')
        print(f'Checking: probabilities - {discard_probabilities.T}')
        print(f'Checking: pos of discard - {discard_argmax}')
        print(f'Checking: p of discard - {discard_probabilities.T[discard_argmax][0]}')
    return p_action
