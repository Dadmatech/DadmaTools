from sklearn.metrics import precision_recall_fscore_support

def score_by_sent(y_true, y_pred, logger):
    assert (len(y_true) == len(y_pred))
    y_true = [i.item() for i in y_true]
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    return {
        'p': prec_micro * 100,
        'r': rec_micro * 100,
        'f1': f1_micro * 100
    }
