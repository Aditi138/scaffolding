"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate --help
    usage: run [command] evaluate [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive-file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation-data-file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
"""
from typing import TextIO, Optional, List, Set, Tuple,Dict, Any
import argparse
import logging

import tqdm
import os
from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment
from allennlp.data import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Evaluate(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
            name, description=description, help='Evaluate the specified model + dataset')

        archive_file = subparser.add_mutually_exclusive_group(required=True)
        archive_file.add_argument(
            '--archive-file', type=str, help='path to an archived trained model')
        archive_file.add_argument(
            '--archive_file', type=str, help=argparse.SUPPRESS)

        evaluation_data_file = subparser.add_mutually_exclusive_group(
            required=True)
        evaluation_data_file.add_argument('--evaluation-data-file',
                                          type=str,
                                          help='path to the file containing the evaluation data')
        evaluation_data_file.add_argument('--evaluation_data_file',
                                          type=str,
                                          help=argparse.SUPPRESS)

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')
        cuda_device.add_argument(
            '--cuda_device', type=int, help=argparse.SUPPRESS)

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate(model: Model,
             dataset: Dataset,
             iterator: DataIterator,
             cuda_device: int) -> Dict[str, Any]:
    model.eval()

    generator = iterator(dataset, num_epochs=1,
                         cuda_device=cuda_device, for_training=False)
    logger.info("Iterating over dataset")
    generator_tqdm = tqdm.tqdm(
        generator, total=iterator.get_num_batches(dataset))
    for batch in generator_tqdm:
        model(**batch)
        metrics = model.get_metrics()
        description = ', '.join(["%s: %.5f" % (name, value)
                                 for name, value in metrics.items() if "overall" in name ]) + " ||"
        generator_tqdm.set_description(description)

    #return model.get_metrics()
    model.get_metrics()
    golds = metrics["gold_spans"]
    predictions = metrics["predicted_spans"]
    assert len(dataset.instances) == len(golds) == len(predictions)
    prediction_file_path = os.path.join("./", "predictions.txt")
    prediction_file = open(prediction_file_path, "w+")
    logger.info("Writing predictions in CoNLL-like format to %s", prediction_file_path)
    print("Writing predictions in CoNLL-like format to %s",prediction_file_path)
    for instance, gold, prediction in tqdm.tqdm(zip(dataset.instances, golds, predictions)):
        fields = instance.fields
        verb_index = None
        frame = None
        if "frame" in fields:
            frame = fields["frame"].tokens[0].text
        gf = None
        if "gf" in fields:
            gf = [g.text for g in fields["gf"].tokens]
        pt = None
        if "pt" in fields:
            pt = [p.text for p in fields["pt"].tokens]
        sentence = [token.text for token in fields["tokens"].tokens]
        gold_tags = convert_spans_to_seq(gold, len(sentence))
        predicted_tags = convert_spans_to_seq(prediction, len(sentence))
        assert len(sentence) == len(gold_tags) == len(predicted_tags)
        write_to_conll_eval_file(prediction_file,verb_index,sentence,predicted_tags,gold_tags,frame, gf, pt)
    return model.get_metrics()

def convert_spans_to_seq(spans: Set[Tuple[int, int, str]], length: int) -> List[str]:
    seq = ["O"] * length
    for span in list(spans):
        start, end, label = span
        if label == "O":
            continue
        for position in range(start, end+1):
            if position >= length:
                continue
            assert position < length
            #assert seq[position] == "O"
            seq[position] = label
    return seq

def write_to_conll_eval_file(prediction_file: TextIO,
                            #  gold_file: TextIO,
                             verb_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str],
                             frame: str = None,
                             gf: List[str] = None,
                             pt: List[str] = None):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.
    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    assert len(sentence) == len(prediction) == len(gold_labels)
    verbs = ["-"] * len(sentence)
    if verb_index:
        verbs[verb_index] = "V-" + sentence[verb_index]
        if frame is not None:
            verbs[verb_index] = frame

    idx = 0
    for word, verb, predicted, gold in zip(sentence, verbs, prediction, gold_labels):
        fields = "{0:<20}\t{1:<10}".format(word, verb)
        if gf:
            fields = "{0:}\t{1:>10}".format(fields, gf[idx])
        if pt:
            fields = "{0:}\t{1:>10}".format(fields, pt[idx])

        # gold_file.write("{}\t{}\n".format(fields, gold))
        prediction_file.write("{0:}\t{1:>20}\t{2:>20}\n".format(fields, gold, predicted))

        idx += 1

    prediction_file.write("\n")
    # gold_file.write("\n")


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger(
        'allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(model.vocab)

    iterator = DataIterator.from_params(config.pop("iterator"))

    metrics = evaluate(model, dataset, iterator, args.cuda_device)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
