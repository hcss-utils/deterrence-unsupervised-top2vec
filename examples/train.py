import json
import typer
from enum import Enum
from pathlib import Path
from top2vec import Top2Vec

app = typer.Typer(name="train", add_completion=False, help="This is a training script.")


class Model(str, Enum):
    DOC2VEC = "doc2vec"
    USE = "universal-sentence-encoder"
    USEM = "universal-sentence-encoder-multilingual"
    DBMC = "distiluse-base-multilingual-cased"


class Speed(str, Enum):
    TEST_LEARN = "test-learn"
    FAST_LEARN = "fast-learn"
    LEARN = "learn"
    DEEP_LEARN = "deep-learn"


def check_parent_exists(path):
    if not path.parent.exists():
        typer.echo(f"The path you've supplied {path} does not exist.")
        raise typer.Exit(code=1)
    return path


@app.command()
def train(
    load: Path = typer.Argument(
        ..., help="Path to a file containing documents for training.", exists=True
    ),
    save: Path = typer.Argument(
        ..., help="Output path for the model.", callback=check_parent_exists
    ),
    embedding_model: Model = typer.Option(
        Model.DOC2VEC,
        help="Which model is used to generate the document and word embeddings.",
    ),
    training_speed: Speed = typer.Option(
        Speed.LEARN, help="How fast the model takes to train"
    ),
    workers: int = typer.Option(
        128, help="The amount of worker threads to be used in training the model"
    ),
):
    """Train Top2Vec algorithm."""
    typer.echo("Loading data...")
    with open(load, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [
        doc.get("fulltext", "")
        for doc in data
        if not doc.get("fulltext", "").startswith("Not available.")
    ]

    speed = training_speed.value
    model = embedding_model.value
    if model == "doc2vec":
        typer.echo(
            f"Training the model with following parameters: {model=}, {speed=}, {workers=}"
        )
        t2v = Top2Vec(
            documents=docs,
            embedding_model=model,
            speed=speed,
            workers=workers,
        )
    else:
        typer.echo(f"Training the model with following parameters: {model=}")
        t2v = Top2Vec(documents=docs, embedding_model=model)
    typer.echo(f"Saving the model to {save}")
    t2v.save(save)


if __name__ == "__main__":
    app()
