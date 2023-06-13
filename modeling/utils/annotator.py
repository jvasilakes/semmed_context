import os
import csv
import curses
import argparse
import warnings

import webdataset as wds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str,
                        help="Path to a tar.gz file containing data.")
    parser.add_argument("outdir", type=str,
                        help="Where to save the annotations.")
    parser.add_argument("task", type=str)
    parser.add_argument("--saved-answers", type=str, default=None,
                        help="Path to previously saved CSV of answers.")
    parser.add_argument("--review", action="store_true", default=False,
                        help="Review previously saved answers.")
    return parser.parse_args()


def main(args):
    ds = wds.WebDataset(args.datafile).shuffle(1000).decode()

    saved_answers = None
    if args.saved_answers is not None:
        with open(args.saved_answers, 'r') as inF:
            reader = csv.reader(inF)
            saved_answers = dict(reader)

    writemode = 'a'
    if args.review is True:
        writemode = 'w'
        if args.saved_answers is None:
            warnings.warn("--review specified but no --saved-answers. Ignoring.")  # noqa
            input("ENTER to continue.")

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, f"{args.task}.csv")
    if os.path.exists(outfile):
        if saved_answers is None:
            warnings.warn(f"Output file {outfile} exists, but was not loaded with --saved-answers. It will be overwritten.")  # noqa
            input("ENTER to continue.")

    answers = curses.wrapper(display, ds, args.outdir, args.task,
                             saved_answers=saved_answers,
                             review=args.review)
    if len(answers) > 0:
        save_and_quit(answers, outfile, writemode=writemode)


def display(stdscr, dataset, outdir, task, saved_answers=None, review=False):
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_RED)  # subject
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_BLUE)  # object
    max_y, max_x = stdscr.getmaxyx()

    if saved_answers is None:
        saved_answers = {}
    answers = {}

    keep = lambda eid, answers: eid not in answers  # noqa
    if review is True:
        answers = saved_answers
        keep = lambda eid, answers: eid in answers  # noqa

    BUFSIZE = 100
    buffer = []
    current_buf_idx = 0
    data_iterator = iter(dataset)
    while True:
        if current_buf_idx < len(buffer):
            example = buffer[current_buf_idx]
        else:
            try:
                example = next(data_iterator)
            except StopIteration:
                break
            if keep(example["__key__"], saved_answers):
                current_buf_idx, buffer = add_to_buffer(
                    example, buffer, current_buf_idx, BUFSIZE)
            else:
                continue

        example_id = example["__key__"]
        stdscr.clear()

        if review is True:
            stdscr.addstr("Review Mode\n\n",
                          curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr("Data file:",
                      curses.A_BOLD | curses.A_UNDERLINE)
        datafile = dataset.pipeline[0].urls[0]
        stdscr.addstr(f" {datafile}/{example_id}\n")
        stdscr.addstr("Task:",
                      curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr(f" {task}\n")
        stdscr.addstr("Annotated:",
                      curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr(f" {len(answers) + len(saved_answers)}\n\n")

        txt = example["json"]["text"]
        _, subj_start, subj_end = example["json"]["subject"]
        _, obj_start, obj_end = example["json"]["object"]
        entity_idxs = [[subj_start, subj_end], [obj_start, obj_end]]
        entity_order = sorted(range(2), key=lambda i: entity_idxs[i][0])

        first_s, first_e = entity_idxs[entity_order[0]]
        scnd_s, scnd_e = entity_idxs[entity_order[1]]
        stdscr.addstr("―" * max_x)
        stdscr.addstr(txt[:first_s])
        stdscr.addstr(txt[first_s:first_e],
                      curses.color_pair(entity_order[0]+1) | curses.A_BOLD)
        stdscr.addstr(txt[first_e:scnd_s])
        stdscr.addstr(txt[scnd_s:scnd_e],
                      curses.color_pair(entity_order[1]+1) | curses.A_BOLD)
        stdscr.addstr(txt[scnd_e:])
        stdscr.addstr('\n' + "―" * max_x)
        stdscr.addstr("\n\n")
        display_annotation_task(stdscr, example, task)

        try:
            prev_answer = answers[example_id]
        except KeyError:
            prev_answer = None
        if prev_answer is not None:
            stdscr.addstr(f"\nPrevious answer: {prev_answer}\n")

        stdscr.addstr("Correct? [y]es / [n]o / [q]uit: ")
        stdscr.refresh()

        while True:
            k = stdscr.getch()
            if chr(k) == 'q':
                stdscr.addstr(chr(k))
                stdscr.refresh()
                return answers
            elif chr(k) in ['y', 'n']:
                stdscr.addstr(chr(k))
                stdscr.refresh()
                k2 = stdscr.getch()
                if k2 == 10:  # Enter/Return key
                    answers[example_id] = chr(k)
                    current_buf_idx = increment_buffer(current_buf_idx, buffer)
                    break
                else:
                    y, x = stdscr.getyx()
                    stdscr.move(y, x-1)
                    continue
            # Skip this example
            elif k == curses.KEY_RIGHT:
                if current_buf_idx < len(buffer):
                    current_buf_idx += 1
                break
            # Go back to the previous example
            elif k == curses.KEY_LEFT:
                current_buf_idx = decrement_buffer(current_buf_idx)
                break
            else:
                continue

    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
    stdscr.clear()
    stdscr.addstr("\n\nAnnotation complete!\n\n",
                  curses.color_pair(3) | curses.A_BOLD)
    stdscr.refresh()
    curses.napms(2000)
    stdscr.getkey()

    return answers


def add_to_buffer(example, buffer, current_buf_idx, max_buffer_size):
    buffer.append(example)
    if len(buffer) > max_buffer_size:
        del buffer[0]
    else:
        current_buf_idx = len(buffer) - 1
    return current_buf_idx, buffer


def increment_buffer(buf_idx, buffer):
    new_buf_idx = buf_idx
    if buf_idx < len(buffer):
        new_buf_idx += 1
    return new_buf_idx


def decrement_buffer(buf_idx):
    new_buf_idx = buf_idx
    if buf_idx > 0:
        new_buf_idx -= 1
    return new_buf_idx


def display_annotation_task(stdscr, example, task):
    subj, _, _ = example["json"]["subject"]
    obj, _, _ = example["json"]["object"]
    pred = example["json"]["labels"]["Predicate"]
    pred = pred.split('_')
    lab = example["json"]["labels"][task]
    if task == "Polarity":
        if lab == "Negative":
            pred = "does not " + ' '.join([t.rstrip('SD') for t in pred])
    elif task == "Certainty":
        if lab == "Uncertain":
            pred = "maybe " + ' '.join(pred)
    stdscr.addstr(subj, curses.color_pair(1) | curses.A_BOLD)
    stdscr.addstr(' ' + pred + ' ', curses.A_BOLD)
    stdscr.addstr(obj, curses.color_pair(2) | curses.A_BOLD)
    stdscr.addstr("\n")


def save_and_quit(answers, outfile, writemode='w'):
    with open(outfile, writemode) as outF:
        writer = csv.writer(outF, delimiter=',')
        print(f"Saving to {outfile}")
        for (example_id, ans) in answers.items():
            writer.writerow([example_id, ans])


if __name__ == "__main__":
    args = parse_args()
    main(args)
