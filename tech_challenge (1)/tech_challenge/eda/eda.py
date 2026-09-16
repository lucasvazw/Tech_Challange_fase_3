from pyspark.sql.functions import (
    col,
    count,
    when,
    round
)

# carregamento da gold
df = spark.table(
    "workspace.default.gold_aluno"
)

print("ANÁLISE EXPLORATÓRIA DOS DADOS")

# visão geral da base
print("\nVISÃO GERAL DA BASE")

total_registros = df.count()
total_colunas = len(df.columns)

print("Total de registros:", total_registros)
print("Total de colunas:", total_colunas)

print("\nColunas disponíveis:")

for coluna in df.columns:
    print("-", coluna)


# distribuição do target
print("\nDISTRIBUIÇÃO DO TARGET")

distribuicao_target = (
    df
    .groupBy("in_alfabetizado")
    .agg(
        count("*").alias("quantidade")
    )
    .withColumn(
        "percentual",
        round(
            col("quantidade") / total_registros * 100,
            2
        )
    )
    .orderBy("in_alfabetizado")
)

display(distribuicao_target)

# valores nulos
print("\nVALORES NULOS")


valores_nulos = df.select([
    count(
        when(
            col(c).isNull(),
            c
        )
    ).alias(c)

    for c in df.columns
])

display(valores_nulos)


# distribuição da presença
print("\nDISTRIBUIÇÃO DA PRESENÇA")

presenca = (
    df
    .groupBy(
        "in_presenca_lp",
        "status_presenca_lp"
    )
    .agg(
        count("*").alias("quantidade")
    )
)

display(presenca)

# fim da EDA
print("EDA FINALIZADA COM SUCESSO!")