# -*- coding: utf-8 -*-
# @FileName : calculation_rl
# @Project  : MAHJONG AI
# @Author   : gangyinglau
# @Time     : 25/4/2021 3:06 AM

def cal_probability_of_action(is_trigger_by_rl, epsilon, discard_argmax, raw_predictions):
    p_action = (1 - epsilon) * raw_predictions[discard_argmax]
    if not is_trigger_by_rl:
        p_action += epsilon / 27
    return p_action
