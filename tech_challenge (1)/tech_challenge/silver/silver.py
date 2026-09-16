from pyspark.sql.functions import col, count, when, trim, upper
from pyspark.sql.types import StringType


def bronze_to_silver(bronze_table, silver_table, tipo=None):

    print("=" * 70)
    print(f"Processando: {bronze_table}")
    print("=" * 70)

    df = spark.table(bronze_table)

    print("Quantidade de registros na Bronze:", df.count())
    print("Quantidade de colunas:", len(df.columns))

    df = df.toDF(*[c.lower() for c in df.columns])

    print("Nomes das colunas padronizados.")

    df = df.dropDuplicates()

    print("Após remover duplicados:", df.count())

    if tipo == "aluno":

        df = df.filter(
            col("in_presenca_lp") == 1
        )

        print(
            "Após filtrar alunos presentes:",
            df.count()
        )

        df = df.filter(
            col("vl_proficiencia_lp").isNotNull()
        )

        print(
            "Após remover proficiência nula:",
            df.count()
        )

    for campo in df.schema.fields:

        if isinstance(campo.dataType, StringType):

            df = df.withColumn(
                campo.name,
                trim(col(campo.name))
            )

    if "sg_uf" in df.columns:

        df = df.withColumn(
            "sg_uf",
            upper(trim(col("sg_uf")))
        )

    print("Campos de texto padronizados.")

    print("\nQuantidade de valores nulos por coluna:")

    resultado_nulos = df.select([
        count(
            when(col(c).isNull(), c)
        ).alias(c)
        for c in df.columns
    ])

    display(resultado_nulos)

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(silver_table)
    )

    print(
        f"\nTabela Silver criada com sucesso: {silver_table}"
    )

    display(df.limit(20))


bronze_to_silver(
    "workspace.default.bronze_aluno",
    "workspace.default.silver_aluno",
    tipo="aluno"
)

bronze_to_silver(
    "workspace.default.bronze_estado",
    "workspace.default.silver_estado"
)

bronze_to_silver(
    "workspace.default.bronze_municipio",
    "workspace.default.silver_municipio"
)

bronze_to_silver(
    "workspace.default.bronze_item",
    "workspace.default.silver_item"
)