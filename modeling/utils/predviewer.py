import re
import json
import random
import argparse
from collections import OrderedDict

from flask import Flask
from flask import request
from dominate import document
import dominate.tags as tags
from matplotlib import cm
from matplotlib import colors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath", type=str,
        help="""Path to JSON lines file containing mask info,
                e.g., output by run.py validate --output_token_masks""")
    parser.add_argument(
        "--num-examples", "-N", type=int, default=1000,
        help="Number of examples to load.")
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Localhost port on which to serve the application.")
    parser.add_argument(
        "--shuffle", default=False, action="store_true",
        help="If specified, shuffle the order of examples.")
    parser.add_argument(
        "--random-state", type=int, default=0,
        help="Set the random seed.")
    return parser.parse_args()


def create_app(args):
    app = Flask(__name__)
    app.config["filename"] = args.filepath
    app.config["data"] = load_data(args.filepath, shuffle=args.shuffle,
                                   random_state=args.random_state)
    app.config["state"] = OrderedDict({
        "collapse wordpiece": True,
        "rm_special_tokens": True,
        "correct": True,
        "incorrect": True,
        "max_examples": args.num_examples,
    })

    # Define and register the label matchers.
    filters = Filters()
    labels = sorted(set([d["json"]["label"] for d in app.config["data"]]))
    for lab in labels:
        filters.register_match_fn("label", lab, name=f"label_{lab}")
        filters.register_match_fn("prediction", lab, name=f"prediction_{lab}")
        app.config["state"][f"label_{lab}"] = True
        app.config["state"][f"prediction_{lab}"] = True

    # The main page
    @app.route("/", methods=("GET", "POST"))
    def view_file():
        state = app.config["state"]
        data = app.config["data"]

        if request.method == "POST":
            print("FILTER")
            print(request.form)
            for (inp, value) in state.items():
                input_val = request.form.get(inp)
                if inp == "max_examples":
                    try:
                        val = int(input_val)
                    except ValueError:
                        val = state[inp]
                else:
                    val = input_val == "on"
                state[inp] = val

        d = document(title="Predictions Viewer")
        d += tags.h1("Predictions Viewer")
        d += tags.p(f"Viewing: {app.config['filename']}")
        d += tags.h3("Filters")
        f = tags.form(method="post")
        for (key, val) in state.items():
            if key == "max_examples":
                continue
            inp = tags.input_(_type="checkbox", _id=key, name=key,
                              checked=state.get(key) or False),
            lab = tags.label(key.title(), for_=key)
            f += inp
            f += lab
            f += tags.br()
        lab = tags.label("Max Examples (-1 for all)", for_=key)
        inp = tags.input_(_type="text", _id="max_examples",
                          name="max_examples",
                          placeholder=str(state.get("max_examples")))
        f += lab
        f += inp
        f += tags.br()
        f += tags.input_(_type="submit", value="Apply")
        f += tags.br()
        f += tags.br()
        d += f

        filtered_data = apply_filters(filters, data, state)
        n_shown = min(int(state.get("max_examples")), len(filtered_data))
        d += tags.p(f"Showing {n_shown}/{len(filtered_data)} examples")
        d += tags.br()

        for (i, example) in enumerate(filtered_data):
            if i == int(state.get("max_examples")):
                break
            d += example2html(
                example, collapse_wordpiece=state.get("collapse wordpiece"),
                rm_special_tokens=state.get("rm_special_tokens")
            )
        return d.render()

    return app


def load_data(filepath, shuffle=False, random_state=0):
    data = [json.loads(line) for line in open(filepath)]
    if shuffle is True:
        random.seed(random_state)
        random.shuffle(data)
    return data


def example2html(example, collapse_wordpiece=False,
                 rm_special_tokens=False):
    docid = example["__key__"]
    gold = example["json"]["label"]
    pred = example["json"]["prediction"]

    head = tags.b(f"Doc ID: {docid} | Gold: '{gold}' | Predicted: '{pred}'")

    if gold == pred:
        txt = " ✓ "
        background_color = "#00ff00"
    else:
        txt = " X "
        background_color = "#ff0000"
    sign = tags.span(txt, _class="highlight",
                     style=f"background-color:{background_color}")

    highlighted_tokens = get_highlighted_tokens(
        example, collapse_wordpiece=collapse_wordpiece,
        rm_special_tokens=rm_special_tokens)
    text = tags.p(highlighted_tokens)
    return tags.div(sign, head, text)


def get_highlighted_tokens(example, collapse_wordpiece=False,
                           rm_special_tokens=False):
    spans = []
    tokens_with_masks = zip(example["json"]["tokens"],
                            example["json"]["token_mask"])
    token_idxs = range(len(example["json"]["tokens"]))
    if collapse_wordpiece is True:
        token_idxs, tokens_with_masks = collapse_wordpiece_tokens(
            tokens_with_masks)

    for (i, (tok, z)) in zip(token_idxs, tokens_with_masks):
        curr_tok_is_subj = curr_tok_is_obj = False
        if i in range(*example["json"]["subject_idxs"]):
            curr_tok_is_subj = True
        if i in range(*example["json"]["object_idxs"]):
            curr_tok_is_obj = True

        if rm_special_tokens is True:
            if is_special_token(tok):
                continue

        if curr_tok_is_subj is True:
            style = "color:blue;font-weight:bold"
            title = "Subject"
        elif curr_tok_is_obj is True:
            style = "color:orange;font-weight:bold"
            title = "Object"
        else:
            color = z2color(z)
            style = f"background-color:{color}"
            title = f"{z:.3f}"
        token = tags.span(f" {tok} ", _class="highlight",
                          style=style, title=title)
        spans.append(token)
    return spans


def is_special_token(token):
    specials = [re.escape(r'[CLS]'),
                re.escape(r'[SEP]'),
                r'\[\s?unused[0-9]+\s?\]']
    for special in specials:
        if re.match(special, token) is not None:
            return True
    return False


def z2color(z):
    # Cap the colormap to make the highlighting more readable.
    norm = colors.Normalize(vmin=0, vmax=1.5)
    return colors.rgb2hex(cm.PuRd(norm(z)))


def collapse_wordpiece_tokens(tokens_with_masks):
    collapsed_idxs = []
    output = []
    current_idx = 0
    current_tok = ''
    current_zs = []
    for (i, (tok, z)) in enumerate(tokens_with_masks):
        if tok.startswith("##"):
            current_tok += tok.lstrip("##")
            current_zs.append(z)
        else:
            if len(current_zs) > 0:
                collapsed_idxs.append(current_idx)
                output.append((current_tok, sum(current_zs) / len(current_zs)))
                current_idx = i
                current_tok = ''
                current_zs = []
            current_tok = tok
            current_zs.append(z)
    collapsed_idxs.append(current_idx)
    output.append((current_tok, sum(current_zs) / len(current_zs)))
    return collapsed_idxs, output


def apply_filters(filters, data, state):
    filtered_data = []
    for d in data:
        filter_results = []
        for (group_name, filter_group) in filters.items():
            group_results = []
            for (key, filt) in filter_group.items():
                if state[key] is True:
                    group_results.append(filt(d))
            filter_results.append(any(group_results))
        if all(filter_results) is True:
            filtered_data.append(d)
    return filtered_data


def register(group, name):
    def assign_name(func):
        func._tag = (group, name)
        return func
    return assign_name


class Filters(object):
    """
    Filters are functions that test a datapoint for a condition.
    Filter functions are organized in groups.
    Generally, filter functions within a group should apply to
    mutually exclusive attributes of the datapoints. E.g., whether
    a datapoint is correctly or incorrectly predicted.
    """

    def __init__(self):
        # Initialize the filters.
        _ = self.filters

    @property
    def filters(self):
        if "_filter_registry" in self.__dict__.keys():
            return self._filter_registry
        else:
            self._filter_registry = {}
            for name in dir(self):
                var = getattr(self, name)
                if hasattr(var, "_tag"):
                    group, fn_name = var._tag
                    if group not in self._filter_registry:
                        self._filter_registry[group] = {}
                    self._filter_registry[group][fn_name] = var
            return self._filter_registry

    def register_match_fn(self, key, label_to_match, name=None):
        """
        `key` is a dict key in a datapoint.
        `label_to_match` is the desired value of datapoint[key].
        That is, the match function will return True if
        datapoint[key] == label_to_match
        """
        reg_fn = register(key, label_to_match)
        match_fn = reg_fn(lambda example: example["json"][key] == label_to_match)  # noqa
        if name is None:
            name = label_to_match
        if key not in self._filter_registry.keys():
            self._filter_registry[key] = {}
        self._filter_registry[key][name] = match_fn

    def __getitem__(self, group, fn_name):
        return self.filters[group][fn_name]

    def __setitem__(self, *args, **kwargs):
        raise AttributeError(f"{self.__class__} does not support item assignment.")  # noqa 

    def keys(self):
        return self.filters.keys()

    def values(self):
        return self.filters.values()

    def items(self):
        return self.filters.items()

    @register("answers", "correct")
    def correct(self, example):
        return example["json"]["label"] == example["json"]["prediction"]

    @register("answers", "incorrect")
    def incorrect(self, example):
        return example["json"]["label"] != example["json"]["prediction"]


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    app.run(port=args.port)
