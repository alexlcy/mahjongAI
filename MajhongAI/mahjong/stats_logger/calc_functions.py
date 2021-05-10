from mahjong.stats_logger.logger import players_dict_logger, logger


@players_dict_logger(data_name='win_times')
def calc_win_times(win_times, game_no):
    return win_times, game_no


@players_dict_logger(data_name='win_rates')
def calc_win_rates(win_times, game_no):
    cal_win_rate = {}
    for i in range(4):
        cal_win_rate[i] = win_times[i] / game_no
    return cal_win_rate, game_no


@players_dict_logger(data_name='hu_scores')
def calc_hu_scores(hu_scores, game_no):
    hu_score_result = {}
    for i in range(4):
        hu_score_result[i] = hu_scores[i]
    return hu_score_result, game_no


@players_dict_logger(data_name='hu_score_each_game')
def calc_hu_score_each_game(hu_rewards, game_no):
    hu_reward_result = {}
    for i in range(4):
        hu_reward_result[i] = hu_rewards[i]
    return hu_reward_result, game_no


@logger(data_name='MSE')
def calc_mean_loss_each_train(loss, no):
    return loss, no
