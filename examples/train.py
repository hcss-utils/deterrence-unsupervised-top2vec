import json
import typer
from pathlib import Path
from top2vec import Top2Vec

app = typer.Typer(name="train", add_completion=False, help="This is a training script.")


def check_parent_exists(path):
    if not path.parent.exists():
        typer.echo(f"The path you've supplied {path} does not exist.")
        raise typer.Exit(code=1)
    return path


def check_embedding_model(model):
    valid_models = (
        "doc2vec",
        "universal-sentence-encoder",
        "universal-sentence-encoder-multilingual",
        "distiluse-base-multilingual-cased",
    )
    model = model.lower().strip()
    if model not in valid_models:
        typer.echo(f"{model=} is not supported.")
        typer.echo(
            f"The valid string options for 'embedding_model' are: {', '.join(valid_models)}"
        )
        raise typer.Exit(code=1)
    return model


def check_speed(speed):
    valid_speeds = ("test-learn", "fast-learn", "learn", "deep-learn")
    speed = speed.lower().strip()
    if speed not in valid_speeds:
        typer.echo(f"{speed=} is not supported.")
        typer.echo(f"speed parameter needs to be one of: {', '.join(valid_speeds[1:])}")
        raise typer.Exit(code=1)
    return speed


@app.command()
def train(
    data: Path = typer.Argument(
        ...,
        help="JSON file containing documents for training.",
        exists=True
    ),
    model: Path = typer.Argument(
        ..., help="Output path for model.", callback=check_parent_exists
    ),
    embedding_model: str = typer.Argument(
        "doc2vec",
        help="Which model is used to generate the document and word embeddings.",
        callback=check_embedding_model,
    ),
    speed: str = typer.Option(
        "learn", help="How fast the model takes to train", callback=check_speed
    ),
    workers: int = typer.Option(
        128, help="The amount of worker threads to be used in training the model"
    ),
    preprocess_data: bool = typer.Option(True, is_flag=True, help="Remove empty docs."),
):
    """Train Top2Vec algorithm."""
    typer.echo("Loading data...")
    with open(data, "r", encoding="utf-8") as f:
        data = json.load(f)
    if preprocess_data:
        docs = [
            doc.get("fulltext", "")
            for doc in data
            if not doc.get("fulltext", "").startswith("Not available.")
        ]
    else:
        docs = [doc.get("fulltext", "") for doc in data]
    if embedding_model == "doc2vec":
        typer.echo(
            f"Training the model with following parameters: {embedding_model=}, {speed=}, {workers=}"
        )
        t2v = Top2Vec(
            documents=docs,
            embedding_model=embedding_model,
            speed=speed,
            workers=workers,
        )
    else:
        typer.echo(f"Training the model with following parameters: {embedding_model=}")
        t2v = Top2Vec(documents=docs, embedding_model=embedding_model)
    typer.echo(f"Saving the model to {model}")
    t2v.save(model)


if __name__ == "__main__":
    app()
