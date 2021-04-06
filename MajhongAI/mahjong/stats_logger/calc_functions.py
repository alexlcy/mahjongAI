from mahjong.stats_logger.logger import players_dict_logger


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
