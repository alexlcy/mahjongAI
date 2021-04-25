# -*- coding: utf-8 -*-
# @FileName : calculation_rl
# @Project  : MAHJONG AI
# @Author   : gangyinglau
# @Time     : 25/4/2021 3:06 AM

def cal_probability_of_action(is_trigger_by_rl, epsilon, discard_argmax, raw_predictions):
    p_action = 0
    try:
        p_action = (1 - epsilon) * raw_predictions.T[discard_argmax][0]
    except Exception:
        print('Error here, type 1: experience/cal_probability_of_action')
    if not is_trigger_by_rl:
        try:
            p_action += epsilon / 27
        except Exception:
            print('Error here, type 4: experience/cal_probability_of_action')
    return p_action
