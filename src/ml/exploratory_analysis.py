"""Funções para análise exploratória dos dados."""


def run_exploratory_analysis(dataframe):
    """Executa a análise exploratória inicial e retorna um resumo."""
    return {
        "shape": dataframe.shape,
        "columns": list(dataframe.columns),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
    }
