SELECT
    ano,
    id_municipio,
    serie,
    rede,
    COUNT(*) AS quantidade
FROM workspace.default.br_inep_avaliacao_alfabetizacao_municipio
WHERE id_municipio IN (
    SELECT id_municipio
    FROM workspace.default.br_inep_avaliacao_alfabetizacao_municipio
    GROUP BY id_municipio, ano
    HAVING COUNT(*) > 1
)
GROUP BY
    ano,
    id_municipio,
    serie,
    rede
ORDER BY
    ano,
    id_municipio,
    rede;