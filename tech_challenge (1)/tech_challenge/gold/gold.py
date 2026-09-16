from pyspark.sql.functions import (
    col,
    when,
    lit,
    round,
    avg,
    count,
    sum,
    lag
)
from pyspark.sql.window import Window


aluno = (
    spark.table("workspace.default.silver_aluno")
    .select(
        "id_aluno",
        "id_escola",
        "nu_ano_avaliacao",
        "co_uf",
        "co_municipio",
        "tp_dependencia",
        "vl_proficiencia_lp",
        "in_alfabetizado",
        "in_presenca_lp"
    )
)


estado = (
    spark.table("workspace.default.silver_estado")
    .select(
        "co_uf",
        "sg_uf"
    )
    .dropDuplicates()
)


municipio = (
    spark.table("workspace.default.silver_municipio")
    .select(
        "co_municipio",
        "no_municipio"
    )
    .dropDuplicates()
)


print("Total de registros na Silver Aluno:", aluno.count())


gold_base = (
    aluno
    .join(
        municipio,
        on="co_municipio",
        how="left"
    )
    .join(
        estado,
        on="co_uf",
        how="left"
    )
)


gold_base = (
    gold_base
    .withColumn(
        "co_municipio",
        when(
            col("co_municipio").isNull(),
            lit(0)
        ).otherwise(col("co_municipio"))
    )
    .withColumn(
        "co_uf",
        when(
            col("co_uf").isNull(),
            lit(0)
        ).otherwise(col("co_uf"))
    )
    .withColumn(
        "tp_dependencia",
        when(
            col("tp_dependencia").isNull(),
            lit(0)
        ).otherwise(col("tp_dependencia"))
    )
    .withColumn(
        "no_municipio",
        when(
            col("no_municipio").isNull() |
            (col("no_municipio") == ""),
            lit("Não Informado")
        ).otherwise(col("no_municipio"))
    )
    .withColumn(
        "sg_uf",
        when(
            col("sg_uf").isNull() |
            (col("sg_uf") == ""),
            lit("Não Informado")
        ).otherwise(col("sg_uf"))
    )
)


print("Total de registros após enriquecimento:", gold_base.count())


gold_aluno = (
    gold_base
    .select(
        "id_aluno",
        "id_escola",
        "nu_ano_avaliacao",
        "co_uf",
        "sg_uf",
        "co_municipio",
        "no_municipio",
        "tp_dependencia",
        "vl_proficiencia_lp",
        "in_alfabetizado",
        "in_presenca_lp"
    )
    .withColumn(
        "status_alfabetizacao",
        when(
            col("in_alfabetizado") == 1,
            "Alfabetizado"
        )
        .when(
            col("in_alfabetizado") == 0,
            "Não Alfabetizado"
        )
        .otherwise("Não Informado")
    )
    .withColumn(
        "status_presenca_lp",
        when(
            col("in_presenca_lp") == 1,
            "Presente"
        )
        .when(
            col("in_presenca_lp") == 0,
            "Ausente"
        )
        .otherwise("Não Informado")
    )
    .withColumn(
        "tp_dependencia_desc",
        when(
            col("tp_dependencia") == 1,
            "Federal"
        )
        .when(
            col("tp_dependencia") == 2,
            "Estadual"
        )
        .when(
            col("tp_dependencia") == 3,
            "Municipal"
        )
        .when(
            col("tp_dependencia") == 4,
            "Privada"
        )
        .otherwise("Não Informado")
    )
)


(
    gold_aluno.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("nu_ano_avaliacao")
    .saveAsTable("workspace.default.gold_aluno")
)


print("gold_aluno criada com sucesso!")


df_municipio_base = (
    gold_base
    .groupBy(
        "nu_ano_avaliacao",
        "co_uf",
        "sg_uf",
        "co_municipio",
        "no_municipio",
        "tp_dependencia"
    )
    .agg(
        count("*").alias("total_alunos"),
        count(
            when(
                col("in_alfabetizado").isNotNull(),
                1
            )
        ).alias("alunos_com_classificacao"),
        round(
            avg("vl_proficiencia_lp"),
            2
        ).alias("media_proficiencia_lp"),
        round(
            sum(
                when(
                    col("in_alfabetizado") == 1,
                    1
                ).otherwise(0)
            )
            /
            count(
                when(
                    col("in_alfabetizado").isNotNull(),
                    1
                )
            )
            * 100,
            2
        ).alias("percentual_alfabetizados"),
        round(
            sum(
                when(
                    col("in_alfabetizado") == 0,
                    1
                ).otherwise(0)
            )
            /
            count(
                when(
                    col("in_alfabetizado").isNotNull(),
                    1
                )
            )
            * 100,
            2
        ).alias("percentual_nao_alfabetizados")
    )
)


window_municipio = (
    Window
    .partitionBy(
        "co_uf",
        "co_municipio",
        "tp_dependencia"
    )
    .orderBy("nu_ano_avaliacao")
)


df_municipio = (
    df_municipio_base
    .withColumn(
        "total_alunos_ano_anterior",
        lag("total_alunos").over(window_municipio)
    )
    .withColumn(
        "media_proficiencia_ano_anterior",
        lag("media_proficiencia_lp").over(window_municipio)
    )
    .withColumn(
        "percentual_alfabetizados_ano_anterior",
        lag("percentual_alfabetizados").over(window_municipio)
    )
    .withColumn(
        "percentual_nao_alfabetizados_ano_anterior",
        lag("percentual_nao_alfabetizados").over(window_municipio)
    )
    .withColumn(
        "variacao_media_proficiencia",
        round(
            col("media_proficiencia_lp") -
            col("media_proficiencia_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_pontos_percentuais_alfabetizados",
        round(
            col("percentual_alfabetizados") -
            col("percentual_alfabetizados_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_pontos_percentuais_nao_alfabetizados",
        round(
            col("percentual_nao_alfabetizados") -
            col("percentual_nao_alfabetizados_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_total_alunos_percentual",
        when(
            col("total_alunos_ano_anterior") > 0,
            round(
                (
                    (
                        col("total_alunos") -
                        col("total_alunos_ano_anterior")
                    )
                    /
                    col("total_alunos_ano_anterior")
                ) * 100,
                2
            )
        )
    )
)


(
    df_municipio.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("nu_ano_avaliacao")
    .saveAsTable(
        "workspace.default.gold_indicador_municipio"
    )
)


print("gold_indicador_municipio criada com sucesso!")


df_estado_base = (
    gold_base
    .groupBy(
        "nu_ano_avaliacao",
        "co_uf",
        "sg_uf",
        "tp_dependencia"
    )
    .agg(
        count("*").alias("total_alunos"),
        count(
            when(
                col("in_alfabetizado").isNotNull(),
                1
            )
        ).alias("alunos_com_classificacao"),
        round(
            avg("vl_proficiencia_lp"),
            2
        ).alias("media_proficiencia_lp"),
        round(
            sum(
                when(
                    col("in_alfabetizado") == 1,
                    1
                ).otherwise(0)
            )
            /
            count(
                when(
                    col("in_alfabetizado").isNotNull(),
                    1
                )
            )
            * 100,
            2
        ).alias("percentual_alfabetizados"),
        round(
            sum(
                when(
                    col("in_alfabetizado") == 0,
                    1
                ).otherwise(0)
            )
            /
            count(
                when(
                    col("in_alfabetizado").isNotNull(),
                    1
                )
            )
            * 100,
            2
        ).alias("percentual_nao_alfabetizados")
    )
)


window_estado = (
    Window
    .partitionBy(
        "co_uf",
        "tp_dependencia"
    )
    .orderBy("nu_ano_avaliacao")
)


df_estado = (
    df_estado_base
    .withColumn(
        "total_alunos_ano_anterior",
        lag("total_alunos").over(window_estado)
    )
    .withColumn(
        "media_proficiencia_ano_anterior",
        lag("media_proficiencia_lp").over(window_estado)
    )
    .withColumn(
        "percentual_alfabetizados_ano_anterior",
        lag("percentual_alfabetizados").over(window_estado)
    )
    .withColumn(
        "percentual_nao_alfabetizados_ano_anterior",
        lag("percentual_nao_alfabetizados").over(window_estado)
    )
    .withColumn(
        "variacao_media_proficiencia",
        round(
            col("media_proficiencia_lp") -
            col("media_proficiencia_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_pontos_percentuais_alfabetizados",
        round(
            col("percentual_alfabetizados") -
            col("percentual_alfabetizados_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_pontos_percentuais_nao_alfabetizados",
        round(
            col("percentual_nao_alfabetizados") -
            col("percentual_nao_alfabetizados_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_total_alunos_percentual",
        when(
            col("total_alunos_ano_anterior") > 0,
            round(
                (
                    (
                        col("total_alunos") -
                        col("total_alunos_ano_anterior")
                    )
                    /
                    col("total_alunos_ano_anterior")
                ) * 100,
                2
            )
        )
    )
)


(
    df_estado.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("nu_ano_avaliacao")
    .saveAsTable(
        "workspace.default.gold_indicador_estado"
    )
)


print("gold_indicador_estado criada com sucesso!")


df_brasil_base = (
    gold_base
    .groupBy(
        "nu_ano_avaliacao",
        "tp_dependencia"
    )
    .agg(
        count("*").alias("total_alunos"),
        count(
            when(
                col("in_alfabetizado").isNotNull(),
                1
            )
        ).alias("alunos_com_classificacao"),
        round(
            avg("vl_proficiencia_lp"),
            2
        ).alias("media_proficiencia_lp"),
        round(
            sum(
                when(
                    col("in_alfabetizado") == 1,
                    1
                ).otherwise(0)
            )
            /
            count(
                when(
                    col("in_alfabetizado").isNotNull(),
                    1
                )
            )
            * 100,
            2
        ).alias("percentual_alfabetizados"),
        round(
            sum(
                when(
                    col("in_alfabetizado") == 0,
                    1
                ).otherwise(0)
            )
            /
            count(
                when(
                    col("in_alfabetizado").isNotNull(),
                    1
                )
            )
            * 100,
            2
        ).alias("percentual_nao_alfabetizados")
    )
)


window_brasil = (
    Window
    .partitionBy("tp_dependencia")
    .orderBy("nu_ano_avaliacao")
)


df_brasil = (
    df_brasil_base
    .withColumn(
        "total_alunos_ano_anterior",
        lag("total_alunos").over(window_brasil)
    )
    .withColumn(
        "media_proficiencia_ano_anterior",
        lag("media_proficiencia_lp").over(window_brasil)
    )
    .withColumn(
        "percentual_alfabetizados_ano_anterior",
        lag("percentual_alfabetizados").over(window_brasil)
    )
    .withColumn(
        "percentual_nao_alfabetizados_ano_anterior",
        lag("percentual_nao_alfabetizados").over(window_brasil)
    )
    .withColumn(
        "variacao_media_proficiencia",
        round(
            col("media_proficiencia_lp") -
            col("media_proficiencia_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_pontos_percentuais_alfabetizados",
        round(
            col("percentual_alfabetizados") -
            col("percentual_alfabetizados_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_pontos_percentuais_nao_alfabetizados",
        round(
            col("percentual_nao_alfabetizados") -
            col("percentual_nao_alfabetizados_ano_anterior"),
            2
        )
    )
    .withColumn(
        "variacao_total_alunos_percentual",
        when(
            col("total_alunos_ano_anterior") > 0,
            round(
                (
                    (
                        col("total_alunos") -
                        col("total_alunos_ano_anterior")
                    )
                    /
                    col("total_alunos_ano_anterior")
                ) * 100,
                2
            )
        )
    )
)


(
    df_brasil.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("nu_ano_avaliacao")
    .saveAsTable(
        "workspace.default.gold_indicador_brasil"
    )
)


print("gold_indicador_brasil criada com sucesso!")


gold_indicador_municipio = (
    df_municipio
    .withColumn("nivel", lit("Município"))
    .select(
        "nu_ano_avaliacao",
        "nivel",
        "co_uf",
        "sg_uf",
        "co_municipio",
        "no_municipio",
        "tp_dependencia",
        "total_alunos",
        "alunos_com_classificacao",
        "media_proficiencia_lp",
        "percentual_alfabetizados",
        "percentual_nao_alfabetizados",
        "total_alunos_ano_anterior",
        "media_proficiencia_ano_anterior",
        "percentual_alfabetizados_ano_anterior",
        "percentual_nao_alfabetizados_ano_anterior",
        "variacao_media_proficiencia",
        "variacao_pontos_percentuais_alfabetizados",
        "variacao_pontos_percentuais_nao_alfabetizados",
        "variacao_total_alunos_percentual"
    )
)


gold_indicador_estado = (
    df_estado
    .withColumn("nivel", lit("Estado"))
    .withColumn("co_municipio", lit(0))
    .withColumn("no_municipio", lit("Não se aplica"))
    .select(
        "nu_ano_avaliacao",
        "nivel",
        "co_uf",
        "sg_uf",
        "co_municipio",
        "no_municipio",
        "tp_dependencia",
        "total_alunos",
        "alunos_com_classificacao",
        "media_proficiencia_lp",
        "percentual_alfabetizados",
        "percentual_nao_alfabetizados",
        "total_alunos_ano_anterior",
        "media_proficiencia_ano_anterior",
        "percentual_alfabetizados_ano_anterior",
        "percentual_nao_alfabetizados_ano_anterior",
        "variacao_media_proficiencia",
        "variacao_pontos_percentuais_alfabetizados",
        "variacao_pontos_percentuais_nao_alfabetizados",
        "variacao_total_alunos_percentual"
    )
)


gold_indicador_brasil = (
    df_brasil
    .withColumn("nivel", lit("Brasil"))
    .withColumn("co_uf", lit(0))
    .withColumn("sg_uf", lit("Brasil"))
    .withColumn("co_municipio", lit(0))
    .withColumn("no_municipio", lit("Brasil"))
    .select(
        "nu_ano_avaliacao",
        "nivel",
        "co_uf",
        "sg_uf",
        "co_municipio",
        "no_municipio",
        "tp_dependencia",
        "total_alunos",
        "alunos_com_classificacao",
        "media_proficiencia_lp",
        "percentual_alfabetizados",
        "percentual_nao_alfabetizados",
        "total_alunos_ano_anterior",
        "media_proficiencia_ano_anterior",
        "percentual_alfabetizados_ano_anterior",
        "percentual_nao_alfabetizados_ano_anterior",
        "variacao_media_proficiencia",
        "variacao_pontos_percentuais_alfabetizados",
        "variacao_pontos_percentuais_nao_alfabetizados",
        "variacao_total_alunos_percentual"
    )
)


gold_indicador = (
    gold_indicador_municipio
    .unionByName(gold_indicador_estado)
    .unionByName(gold_indicador_brasil)
)


(
    gold_indicador.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("nu_ano_avaliacao")
    .saveAsTable(
        "workspace.default.gold_indicador"
    )
)


print("gold_indicador criada com sucesso!")


gold_referencia_municipio = (
    spark.table(
        "workspace.default.br_inep_avaliacao_alfabetizacao_municipio"
    )
    .select(
        "ano",
        "id_municipio",
        "serie",
        "rede",
        "taxa_alfabetizacao",
        "media_portugues",
        "proporcao_aluno_nivel_0",
        "proporcao_aluno_nivel_1",
        "proporcao_aluno_nivel_2",
        "proporcao_aluno_nivel_3",
        "proporcao_aluno_nivel_4",
        "proporcao_aluno_nivel_5",
        "proporcao_aluno_nivel_6",
        "proporcao_aluno_nivel_7",
        "proporcao_aluno_nivel_8"
    )
)


(
    gold_referencia_municipio.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("ano")
    .saveAsTable(
        "workspace.default.gold_referencia_municipio"
    )
)


print("gold_referencia_municipio criada com sucesso!")


print("=" * 70)
print("GOLD FINALIZADA COM SUCESSO!")
print("=" * 70)


print("\nGold Aluno:")
display(
    spark.table(
        "workspace.default.gold_aluno"
    ).limit(20)
)


print("\nGold Indicador:")
display(
    spark.table(
        "workspace.default.gold_indicador"
    ).limit(20)
)


print("\nGold Indicador Município:")
display(
    spark.table(
        "workspace.default.gold_indicador_municipio"
    ).limit(20)
)


print("\nGold Indicador Estado:")
display(
    spark.table(
        "workspace.default.gold_indicador_estado"
    ).limit(20)
)


print("\nGold Indicador Brasil:")
display(
    spark.table(
        "workspace.default.gold_indicador_brasil"
    ).limit(20)
)


print("\nGold Referência Município:")
display(
    spark.table(
        "workspace.default.gold_referencia_municipio"
    ).limit(20)
)