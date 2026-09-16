from pyspark.sql.functions import current_timestamp, lit

anos = [24, 25]

entidades = [
    "aluno",
    "municipio",
    "estado",
    "item",
]


def criar_bronze(entidade, anos, origem="BigQuery"):

    tabela_destino = f"workspace.default.bronze_{entidade}"

    print("=" * 70)
    print(f"Processando: {entidade}")
    print("=" * 70)

    # Remove a tabela anterior para permitir reprocessamento
    if spark.catalog.tableExists(tabela_destino):
        spark.sql(f"DROP TABLE {tabela_destino}")

    dataframes = []

    for ano in anos:

        tabela_origem = f"workspace.default.df_{entidade}_{ano}"

        df = (
            spark.table(tabela_origem)
            .withColumn("dt_ingestao", current_timestamp())
            .withColumn("origem", lit(origem))
        )

        dataframes.append(df)

        print(f"Origem carregada: {tabela_origem}")
        print(f"Registros: {df.count()}")

    # Junta todos os anos
    df_final = dataframes[0]

    for df in dataframes[1:]:
        df_final = df_final.unionByName(
            df,
            allowMissingColumns=True
        )

    print(f"Total de registros: {df_final.count()}")

    # Grava uma única tabela Delta
    (
        df_final.write
        .format("delta")
        .option("mergeSchema", "true")
        .mode("overwrite")
        .saveAsTable(tabela_destino)
    )

    print(f"Tabela criada: {tabela_destino}")


for entidade in entidades:
    criar_bronze(entidade, anos)

print("=" * 70)
print("TODAS AS TABELAS BRONZE FORAM CRIADAS COM SUCESSO!")
print("=" * 70)