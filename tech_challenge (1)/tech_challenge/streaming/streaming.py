from pyspark.sql import functions as F


TABELA_ORIGEM = "workspace.default.eventos_streaming_origem"
TABELA_DESTINO = "workspace.default.bronze_streaming"


# Cria um Volume para guardar o checkpoint
spark.sql("""
    CREATE VOLUME IF NOT EXISTS
    workspace.default.checkpoints_streaming
""")


CHECKPOINT = (
    "/Volumes/workspace/default/"
    "checkpoints_streaming/alfabetizacao"
)


# Busca um município real da camada Gold
municipio = (
    spark.table(
        "workspace.default.gold_indicador_municipio"
    )
    .select(
        "nu_ano_avaliacao",
        "co_uf",
        "sg_uf",
        "co_municipio",
        "no_municipio",
        "tp_dependencia"
    )
    .dropDuplicates()
    .first()
)


if municipio is None:
    raise ValueError(
        "Nenhum município encontrado na camada Gold."
    )


# Simula 30 novas medições
novos_eventos = (
    spark.range(30)

    .withColumn(
        "id_evento",
        F.expr("uuid()")
    )

    .withColumn(
        "dt_evento",
        F.current_timestamp()
    )

    .withColumn(
        "nu_ano_avaliacao",
        F.lit(municipio["nu_ano_avaliacao"])
    )

    .withColumn(
        "co_uf",
        F.lit(municipio["co_uf"])
    )

    .withColumn(
        "sg_uf",
        F.lit(municipio["sg_uf"])
    )

    .withColumn(
        "co_municipio",
        F.lit(municipio["co_municipio"])
    )

    .withColumn(
        "no_municipio",
        F.lit(municipio["no_municipio"])
    )

    .withColumn(
        "tp_dependencia",
        F.lit(municipio["tp_dependencia"])
    )

    .withColumn(
        "in_presenca_lp",
        F.lit(1)
    )

    .withColumn(
        "vl_proficiencia_lp",
        (
            F.lit(730) + F.col("id")
        ).cast("double")
    )

    .withColumn(
        "in_alfabetizado",
        F.when(
            F.col("vl_proficiencia_lp") >= 743,
            1
        ).otherwise(0)
    )

    .withColumn(
        "vl_peso_aluno_lp",
        F.lit(1.0)
    )

    .withColumn(
        "tipo_evento",
        F.lit("nova_medicao_desempenho")
    )

    .select(
        "id_evento",
        "dt_evento",
        "tipo_evento",
        "nu_ano_avaliacao",
        "co_uf",
        "sg_uf",
        "co_municipio",
        "no_municipio",
        "tp_dependencia",
        "in_presenca_lp",
        "vl_proficiencia_lp",
        "in_alfabetizado",
        "vl_peso_aluno_lp"
    )
)


# Adiciona os eventos na tabela de origem
(
    novos_eventos.write
    .format("delta")
    .mode("append")
    .saveAsTable(TABELA_ORIGEM)
)


# Lê os novos registros como streaming
eventos_streaming = (
    spark.readStream
    .table(TABELA_ORIGEM)
    .withColumn(
        "dt_ingestao",
        F.current_timestamp()
    )
    .withColumn(
        "origem",
        F.lit("simulacao_streaming")
    )
)


# Salva os eventos na Bronze Streaming
consulta = (
    eventos_streaming.writeStream
    .format("delta")
    .outputMode("append")
    .option(
        "checkpointLocation",
        CHECKPOINT
    )
    .queryName(
        "streaming_alfabetizacao"
    )
    .trigger(
        availableNow=True
    )
    .toTable(TABELA_DESTINO)
)


print("✅ Streaming iniciado!")


consulta.awaitTermination()


print("✅ Streaming finalizado com sucesso!")


total_eventos = (
    spark.table(TABELA_DESTINO)
    .count()
)


print(
    f"✅ Total de eventos armazenados: "
    f"{total_eventos:,}"
)


display(
    spark.table(TABELA_DESTINO)
    .orderBy(
        F.col("dt_evento").desc()
    )
    .limit(30)
)