# you may change the polarity in the experiments
POLARITY = 'positive'  # 'negative'
polarity = POLARITY + '_polarity/'

real_T_file_ = 'dataset_op_spam_v1.4/' + polarity + 'truthful_from_TripAdvisor_/' # the path for trustworthy reviews in PREPROCESSED .txt format
real_U_file_ = 'dataset_op_spam_v1.4/' + polarity + 'deceptive_from_MTurk_/' # the path for untrustworthy reviews in PREPROCESSED .txt format