from pykeen.datasets import Nations
from pykeen.models import TransE
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator


def custom_pipeline():
    dataset = Nations()
    training_triples_factory = dataset.training
    # Pick a model
    model = TransE(triples_factory=training_triples_factory)
    # Pick an optimizer from Torch
    optimizer = Adam(params=model.get_grad_params())
    # Pick a training approach (sLCWA or LCWA)
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training_triples_factory,
        optimizer=optimizer,
    )
    # Train like Cristiano Ronaldo
    _ = training_loop.train(
        triples_factory=training_triples_factory,
        num_epochs=5,
        batch_size=256,
    )
    # Pick an evaluator
    evaluator = RankBasedEvaluator()
    # Get triples to test
    mapped_triples = dataset.testing.mapped_triples
    # Evaluate
    results = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        batch_size=1024,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    print(results.to_dict())


if __name__ == '__main__':
    custom_pipeline()