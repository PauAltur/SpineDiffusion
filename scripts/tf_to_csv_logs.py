import argparse
import glob
import os
from pathlib import Path

import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm import tqdm


def main(log_dir: Path):
    """Transforms tf events files to csv files and saves
    them in the same directory as the tf events files.

    Args:
        log_dir (_type_): _description_
    """

    tf_events_files = tqdm(
        glob.iglob(str(log_dir / "**" / "events.out.tfevents.*"), recursive=True),
        desc="Searching for tf events files",
    )

    for tf_events_file in tqdm(
        tf_events_files, desc="Transforming tf events files to csv"
    ):
        df = pd.DataFrame(columns=["time", "tag", "value"])
        for event in summary_iterator(tf_events_file):
            for value in event.summary.value:
                df.loc[len(df)] = [event.wall_time, value.tag, value.simple_value]

        log_dir = Path(tf_events_file).parent
        df.to_csv(f"{log_dir}/events.csv", index=False)
        print(f"Saved {log_dir}/events.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    args = parser.parse_args()
    main(log_dir=Path(args.logdir))
