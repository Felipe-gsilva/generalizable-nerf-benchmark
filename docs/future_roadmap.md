# Roadmap de Melhorias: Few-Shot NeRF e Aprofundamento do Artigo

Este documento consolida as estratégias discutidas para contornar a falta de convergência no regime *Few-Shot* ($N=3, 6, 10$) e melhorar o arcabouço teórico do artigo.

## 1. Melhorias Estruturais no Texto do Paper

A principal contribuição teórica dos testes mais recentes é o **Resolution Mismatch**. O artigo deve ser expandido para evidenciar por que a métrica quantitativa pode mentir.

*   **Subseção Específica (O Fenômeno do Scale Mismatch):** Incluir uma discussão sobre a queda de 26 dB (treino e teste em ds=8) para 12 dB (treino ds=8, teste ds=4) no modelo Nerfacto. Explicar como a baixa resolução atua como uma barreira passa-baixa artificial que camufla a falta de coerência geométrica 3D do modelo quando dados escassos são fornecidos.
    *   **Adoção de Análise Visual e Profundidade:** Como as métricas PSNR puras são insuficientes no regime few-shot, incluir *Depth Maps* . O GNT, por utilizar atenção transversal nas *epipolar lines*, tenderá a gerar um mapa de profundidade muito mais sólido do que os artefatos em forma de "fumaça/floaters" do Instant-NGP.

## 2. Abordagens Experimentais (O Que Testar)

Se os modelos atuais (especialmente per-scene) não convergiram como esperado, as seguintes intervenções técnicas são indicadas:

### A. Treinamento Nativo em Alta Resolução (Remover o Mismatch)
*   **Ação:** Treinar o Nerfacto e Instant-NGP *nativamente* em `downscale_factor=2` ou `downscale_factor=4`.
*   **Expectativa:** O PSNR quantitativo provavelmente será ruim, mas isso *provará empiricamente* na tabela de resultados que, ao ser forçado a renderizar detalhes em alta resolução sem informações de contexto suficientes, modelos per-scene falham miseravelmente, enchendo o espaço vazio de geometria ruidosa (*floaters*).

### B. Ajuste de Hiperparâmetros (Learning Rate no TTA)
*   **Ação:** Ao executar o Test-Time Adaptation (TTA) para o GNT e o PixelNeRF, o Learning Rate padrão pode estar alto demais, gerando *Catastrophic Forgetting* da distribuição pré-treinada.
*   **Expectativa:** Experimentar aplicar *warm-up* na taxa de aprendizado ou reduzir estaticamente o Learning Rate base (de $10^{-3}$ para $10^{-4}$). Adicionalmente, no caso do Nerfacto, limitar o tamanho da tabela de *Hash* para restringir o *overfitting* espacial às três imagens exatas.

### C. Inserção de Regularização (O Caminho do Estado da Arte)
A chave de sucesso para NeRFs com poucas vistas na literatura moderna é forçar a geometria do espaço vazio. Sem restrições de geometria, o espaço 3D é subdeterminado.
*   **Ação (Depth Priors):** Extrair pseudoprofundidade das 3 imagens de treino (via redes monocular pré-treinadas como MiDaS ou DPT) e inserir uma *Depth Loss* durante o treinamento do Nerfacto.
*   **Ação (FreeNeRF / Frequency Masking):** Inserir um mascaramento progressivo no *Positional Encoding*. Nos primeiros $500$ steps de treinamento, o modelo só consegue "enxergar" as frequências baixas das imagens de treino (coisas grandes como a estrutura geral do objeto). Nas etapas finais, as altas frequências (texturas) são liberadas. Isso elimina massivamente os *floaters* nas cenas few-shot.

## 3. Experimentos de Curto Prazo (Executáveis em ~3 Horas)

A fim de alavancar o PSNR com pouco esforço de reengenharia, as seguintes possibilidades foram elaboradas:

1.  **Avaliação Pura em Downscale 2:** Executar todos os testes de 10 vistas com `downscale_factor=2` ao invés de 4. Como haverá 4x mais informação em pixels na imagem, o "teto" de qualidade alcançável pelos modelos sobe substancialmente, contanto que se tenha VRAM suficiente na GTX 1660.
