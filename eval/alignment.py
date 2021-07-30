import logging


def align_with_ground_truth(docid, predicted, ground_truth, max_look_ahead=9):
    """Align the predicted tokens with the ground truth tokens.

    This is a greedy heuristic: whenever there is a mismatch, it skips
    over tokens until the sequences match again. This happens to work
    on the turku-one test test but is in no way general. It might
    throw lots of exceptions on another data set with different corner
    cases."""

    # A proper sequence alignment algorithm might be a good idea. I
    # tried python-alignment but it ended up with too deep recursion
    # (maybe it's not suitable for large vocabulary?).

    logging.debug(f'Aligning predicted with ground truth on document {docid}')

    i = 0 # ground truth index
    j = 0 # predicted index
    aligned = []
    while i < len(ground_truth):
        if j >= len(predicted):
            # Probably this simple heuristic has failed. Try to
            # recover by outputting the remaining ground truths with
            # the latest predicted label.
            k = len(ground_truth) - i

            logging.warning(f'possible alignment heuristic mismatch '
                            f'for the last {k} tokens on document {docid}')

            continuation_label = continue_entity_label(predicted[-1][1])
            for gt in ground_truth[i:]:
                aligned.append([gt[0], continuation_label])

            logging.debug(f'{[x[0] for x in ground_truth[i:]]} - <none>')

            i += k

        elif ground_truth[i][0] == predicted[j][0]:
            logging.debug(f'{ground_truth[i][0]} - {predicted[j][0]}')
            
            aligned.append([ground_truth[i][0], predicted[j][1]])
            i += 1
            j += 1

        elif predicted[j][0].startswith(ground_truth[i][0]):
            # Let's assume the ground truth has multiple tokens
            # corresponding to one predicted token.
            k = consume_matches(predicted[j][0], [x[0] for x in ground_truth[i:]])
            if k is None:
                next_gt_tokens = [x[0] for x in ground_truth[i+2:i+max_look_ahead]]
                if j >= len(predicted) - 1:
                    k = len(ground_truth) - i
                else:
                    k = index_is_start_of(next_gt_tokens, predicted[j+1][0]) + 2

            predicted_label = predicted[j][1]
            continuation_label = continue_entity_label(predicted_label)

            aligned.append([ground_truth[i][0], predicted_label])
            for m in range(1, k):
                aligned.append([ground_truth[i+m][0], continuation_label])

            logging.debug(f'{[x[0] for x in ground_truth[i:i+k]]} - {predicted[j][0]}')

            i += k
            j += 1

        else:
            # predicted has multiple tokens corresponding to one
            # ground truth token.
            next_predicted_tokens = [x[0] for x in predicted[j+2:j+max_look_ahead]]
            k = index_is_start_of(next_predicted_tokens, ground_truth[i+1][0]) + 2

            predicted_label = predicted[j][1]
            aligned.append([ground_truth[i][0], predicted_label])

            skipped = predicted[j+1:j+k]
            if not label_continues_or_empty([x[1] for x in skipped], predicted_label):
                logging.warning(f'Discarding predicted entity labels on document {docid}')
                logging.warning(skipped)

            logging.debug(f'{ground_truth[i][0]} - {[x[0] for x in predicted[j:j+k]]}')

            i += 1
            j += k

    # Checking the consistency of the entity labels (I-tag can only
    # follow the corresponding B-tag) in the output would be a good
    # idea except that the input prediction is often inconsistent.

    return aligned


def consume_matches(text, seq):
    i = 0
    textpos = 0
    while i < len(seq) and textpos < len(text):
        if text[textpos:].startswith(seq[i]):
            textpos += len(seq[i])
            i += 1
        else:
            break

    if textpos >= len(text) - 1:
        return i
    else:
        return None


def merge_ground_truth(docid, predicted, ground_truth):
    """Merge predicted and ground truth labels into a combined array.

    The columns of the output are: token, ground truth entity,
    predicted entity."""

    predicted = align_with_ground_truth(docid, predicted, ground_truth)
    assert len(predicted) == len(ground_truth)

    res = []
    for pred, gt in zip(predicted, ground_truth):
        assert pred[0] == gt[0]

        res.append([gt[0], gt[1], pred[1]])

    return res


def index_is_start_of(arr, key):
    for i, v in enumerate(arr):
        if key.startswith(v):
            return i

    raise ValueError(f"No prefix of '{key}' in list")


def label_continues_or_empty(seq, previous_label):
    """Returns True if seq continues previous_label or 'O' (that is no new
    tags starting)."""

    seq = list(seq) # copy

    if previous_label.startswith('B-'):
        continuation_label = 'I-' + previous_label[2:]
    elif previous_label.startswith('I-'):
        continuation_label = previous_label
    else:
        continuation_label = 'O'

    while seq and seq[0] == continuation_label:
        seq.pop(0)

    return all(x == 'O' for x in seq)


def continue_entity_label(entity_label):
    if entity_label.startswith('B-'):
        return 'I-' + entity_label[2:]
    else:
        return entity_label
