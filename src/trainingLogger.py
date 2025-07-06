from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TimeRemainingColumn, BarColumn, TextColumn

class TrainingLogger:
    def __init__(self, log_file=None):
        self.console = Console()
        self.log_file = log_file
        if log_file:
            self.file_console = Console(file=open(log_file, 'w'))
        else:
            self.file_console = None

    def log(self, message):
        self.console.log(message)
        if self.file_console:
            self.file_console.log(message)

    def log_metrics(self, epoch, total_epochs, losses: dict, lr, elapsed):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Epoch", justify="right")
        table.add_column("LR", justify="right")
        table.add_column("Time (s)", justify="right")
        for key in losses.keys():
            table.add_column(key, justify="right")

        table.add_row(
            f"{epoch}/{total_epochs}",
            f"{lr:.2e}",
            f"{elapsed:.2f}",
            *[f"{v:.4e}" for v in losses.values()]
        )

        self.console.print(table)
        if self.file_console:
            self.file_console.print(table)

    def close(self):
        if self.file_console:
            self.file_console.file.close()
