

library(dplyr)
library(ggplot2)
library(broom)
library(sandwich)
library(lmtest)
library(boot)
library(tidyr)
library(scales)
library(ggrepel)
library(mgcv)       # GAM / LOESS
library(patchwork)  # combinar ggplots

# ── 0. CONFIGURACIÓN GLOBAL ───────────────────────────────────────────────────

DATA_PATH         <- "C:/Users/RAYANs/Desktop/proyecto/pricing_clean/src/data/results/ab_results.csv"
OUTPUT_DIR        <- "C:/Users/RAYANs/Desktop/proyecto/pricing_clean/src/data/results/"

GLOBAL_ELASTICITY  <- -1.2   # prior econométrico
SHRINKAGE_BASE     <- 150    # fuerza máxima del shrinkage (categorías sin señal)
SHRINKAGE_STRONG   <- 30     # fuerza mínima (categorías con señal fuerte)
N_BOOT             <- 999
BIN_SD             <- 1.2    # SD del jitter en Ruta B (en pp de descuento)
N_MIN_BIN          <- 30     # mínimo de obs por bin en Ruta C
set.seed(42)

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── 1. CARGA Y PREPARACIÓN DE DATOS ──────────────────────────────────────────

df <- read.csv(DATA_PATH)

df_valid <- df %>%
  filter(discount_applied > 0) %>%
  mutate(
    log_price_ratio = log(1 - discount_applied / 100),
    log_q           = log1p(units_sold)
  )


cat(" RESUMEN DE DATOS\n")

cat(sprintf("  Observaciones totales    : %d\n", nrow(df)))
cat(sprintf("  Observaciones (d > 0)    : %d\n", nrow(df_valid)))
cat(sprintf("  Categorías               : %d\n", n_distinct(df_valid$product_category)))
cat(sprintf("  Niveles de descuento     : %s\n",
            paste(sort(unique(df$discount_applied)), collapse = ", ")))
cat("====================================================================\n\n")

# ── 2. KRUSKAL-WALLIS — señal estadística por categoría ─────────────────────

kw_results <- df %>%
  group_by(product_category) %>%
  summarise(
    p_kw  = kruskal.test(units_sold ~ factor(discount_applied))$p.value,
    n     = n(),
    .groups = "drop"
  ) %>%
  mutate(
    p_kw_adj      = p.adjust(p_kw, method = "BH"),
    señal_kw      = p_kw < 0.05 & n >= 50
  )

# ── 3. TEST DE BALANCE ────────────────────────────────────────────────────────

cat("====================================================================\n")
cat(" TEST DE BALANCE (Chi-cuadrado)\n")
cat("====================================================================\n")
chi_test <- chisq.test(table(df$product_category, df$discount_applied))
cat(sprintf("  Chi²=%.2f  df=%d  p=%.4f\n",
            chi_test$statistic, chi_test$parameter, chi_test$p.value))
if (chi_test$p.value > 0.05) {
  cat("  ✓ Asignación balanceada\n")
} else {
  cat("  ⚠ Posible desbalance — revisar diseño del A/B\n")
}
cat("====================================================================\n\n")




cat(" RUTA A — PRECIO CONTINUO (nivel transacción)\n")


# A1. Modelo OLS con interacción categoría × precio
modelo_ols_int <- lm(log_q ~ log_price_ratio * product_category, data = df_valid)
ct_int         <- coeftest(modelo_ols_int, vcov = vcovHC(modelo_ols_int, "HC3"))

cat_levels <- sort(unique(df_valid$product_category))
ref_cat    <- cat_levels[1]

# Elasticidades: β_precio + β_interacción (=0 para categoría de referencia)
int_names  <- paste0("log_price_ratio:product_category", cat_levels[-1])
int_coefs  <- coef(modelo_ols_int)

elasticidades_A <- tibble(
  product_category = cat_levels,
  elasticity_A = c(
    int_coefs["log_price_ratio"],
    int_coefs["log_price_ratio"] +
      ifelse(int_names %in% names(int_coefs), int_coefs[int_names], 0)
  )
)

cat("\n  Elasticidades OLS (interacción categoría × precio):\n")
print(elasticidades_A %>% arrange(elasticity_A) %>%
        mutate(elasticity_A = round(elasticity_A, 3)), n = 20)

# A2. GAM por categoría — detecta no-linealidad
gam_results_A <- df_valid %>%
  group_by(product_category) %>%
  group_modify(function(data, key) {
    if (nrow(data) < 20 || n_distinct(data$log_price_ratio) < 3) {
      return(tibble(edf = NA, p_nonlinear = NA, gam_r2 = NA, gam_ok = FALSE))
    }
    k_val <- min(4, n_distinct(data$log_price_ratio))
    g <- tryCatch(
      gam(log_q ~ s(log_price_ratio, k = k_val), data = data, method = "REML"),
      error = function(e) NULL
    )
    if (is.null(g)) return(tibble(edf=NA, p_nonlinear=NA, gam_r2=NA, gam_ok=FALSE))
    sg <- summary(g)
    tibble(
      edf         = round(sg$edf, 2),
      p_nonlinear = round(sg$s.pv, 4),
      gam_r2      = round(sg$r.sq, 3),
      gam_ok      = TRUE
    )
  }) %>%
  ungroup()

cat("\n  Test de no-linealidad GAM (edf ≈ 1 → log-lineal OK):\n")
print(gam_results_A %>% arrange(desc(edf)), n = 20)


cat(" RUTA B — SIMULACIÓN DE PRECIO CONTINUO (jitter)\n")


# B1. Añadir jitter calibrado (±BIN_SD pp, sin salir del bin)
bin_half <- 2.5  # mitad del ancho del bin más estrecho

df_sim <- df_valid %>%
  mutate(
    d_jitter = discount_applied + rnorm(n(), 0, BIN_SD),
    d_jitter = pmax(discount_applied - bin_half + 0.1,
                    pmin(discount_applied + bin_half - 0.1, d_jitter)),
    d_jitter        = pmax(d_jitter, 0.5),
    log_price_ratio_sim = log(1 - d_jitter / 100)
  )

# B2. Verificar que el jitter preserva la señal
check_jitter <- df_sim %>%
  group_by(product_category) %>%
  summarise(
    cor_orig   = round(cor(log_price_ratio, log_q), 3),
    cor_jitter = round(cor(log_price_ratio_sim, log_q), 3),
    delta_cor  = round(cor_jitter - cor_orig, 3),
    .groups    = "drop"
  )
cat("\n  Verificación jitter (correlación antes/después):\n")
print(check_jitter, n = 20)

# B3. Regresión con precio simulado
resultados_B_base <- df_sim %>%
  group_by(product_category) %>%
  group_modify(function(data, key) {
    if (nrow(data) < 10) return(tibble(elasticity_B=NA, se_B=NA, r2_B=NA, n=nrow(data)))
    m <- lm(log_q ~ log_price_ratio_sim, data = data)
    s <- summary(m)
    tibble(
      elasticity_B = coef(m)[2],
      se_B         = s$coefficients[2, 2],
      r2_B         = s$r.squared,
      n            = nrow(data)
    )
  }) %>%
  ungroup()

# B4. Bootstrap que propaga incertidumbre del jitter
boot_sim_fn <- function(cat_data, n_boot = 500) {
  replicate(n_boot, {
    d_j <- cat_data$discount_applied + rnorm(nrow(cat_data), 0, BIN_SD)
    d_j <- pmax(cat_data$discount_applied - bin_half + 0.1,
                pmin(cat_data$discount_applied + bin_half - 0.1, d_j))
    d_j <- pmax(d_j, 0.5)
    lpr <- log(1 - d_j / 100)
    lq  <- log1p(cat_data$units_sold)
    idx <- sample(nrow(cat_data), replace = TRUE)
    if (var(lpr[idx]) < 1e-10) return(NA_real_)
    coef(lm(lq[idx] ~ lpr[idx]))[2]
  })
}

ic_B <- df_sim %>%
  group_by(product_category) %>%
  group_modify(function(data, key) {
    if (nrow(data) < 10) return(tibble(ci_low_B=NA, ci_high_B=NA))
    reps <- boot_sim_fn(data, n_boot = 500)
    tibble(
      ci_low_B  = quantile(reps, 0.025, na.rm = TRUE),
      ci_high_B = quantile(reps, 0.975, na.rm = TRUE)
    )
  }) %>%
  ungroup()

resultados_B <- left_join(resultados_B_base, ic_B, by = "product_category")

cat("\n  Elasticidades con precio simulado + IC bootstrap:\n")
print(resultados_B %>% arrange(elasticity_B) %>%
        mutate(across(c(elasticity_B, se_B, ci_low_B, ci_high_B, r2_B), round, 3)), n = 20)



cat(" RUTA C — BINS (WLS + LOESS)\n")

# C1. Bins por cuantil específicos por categoría
make_quantile_bins <- function(x, n_min = N_MIN_BIN) {
  n_bins <- max(4L, min(20L, floor(length(x) / n_min)))
  brks   <- unique(quantile(x, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  if (length(brks) < 3) brks <- c(min(x) - 0.1, median(x), max(x) + 0.1)
  brks
}

df_binned <- df_valid %>%
  group_by(product_category) %>%
  mutate(
    bin_breaks = list(make_quantile_bins(discount_applied, N_MIN_BIN))
  ) %>%
  rowwise() %>%
  mutate(
    bin_label = {
      brks <- unlist(bin_breaks)
      cut(discount_applied, breaks = brks, include.lowest = TRUE, right = FALSE,
          labels = FALSE)
    },
    bin_lo = unlist(bin_breaks)[bin_label],
    bin_hi = unlist(bin_breaks)[bin_label + 1],
    bin_midpoint = (bin_lo + bin_hi) / 2
  ) %>%
  ungroup() %>%
  select(-bin_breaks, -bin_lo, -bin_hi)

# C2. Agregación por (categoría, bin)
agg_C <- df_binned %>%
  group_by(product_category, bin_label, bin_midpoint) %>%
  summarise(
    log_q_med       = median(log_q),
    log_price_ratio = log(1 - median(bin_midpoint, na.rm = TRUE) / 100),
    n_obs           = n(),
    q25             = quantile(log_q, 0.25),
    q75             = quantile(log_q, 0.75),
    .groups         = "drop"
  ) %>%
  filter(!is.na(bin_midpoint), n_obs >= 5, is.finite(log_price_ratio))

cat(sprintf("\n  Bins totales generados: %d (media %.1f por categoría)\n",
            nrow(agg_C),
            nrow(agg_C) / n_distinct(agg_C$product_category)))

# C3. WLS por categoría (ponderado por sqrt(n_obs)) + LOESS
resultados_C <- agg_C %>%
  group_by(product_category) %>%
  group_modify(function(data, key) {
    n_bins <- nrow(data)
    if (n_bins < 3) return(tibble(
      elasticity_C=NA, se_C=NA, r2_ols=NA, r2_wls=NA,
      r2_loess=NA, n_bins=n_bins, curvatura=FALSE
    ))

    m_ols <- lm(log_q_med ~ log_price_ratio, data = data)
    m_wls <- lm(log_q_med ~ log_price_ratio, data = data,
                weights = sqrt(data$n_obs))

    r2_loess <- NA_real_
    curvatura <- FALSE
    if (n_bins >= 5) {
      span_val <- min(0.85, max(0.4, 5 / n_bins))
      lo <- tryCatch(
        loess(log_q_med ~ log_price_ratio, data = data,
              span = span_val, degree = 1, weights = sqrt(data$n_obs)),
        error = function(e) NULL
      )
      if (!is.null(lo)) {
        ss_res   <- sum((data$log_q_med - fitted(lo))^2)
        ss_tot   <- sum((data$log_q_med - mean(data$log_q_med))^2)
        r2_loess <- ifelse(ss_tot > 0, 1 - ss_res / ss_tot, NA)
        curvatura <- !is.na(r2_loess) &&
                     (r2_loess - summary(m_ols)$r.squared) > 0.15
      }
    }

    tibble(
      elasticity_C = coef(m_wls)[2],
      se_C         = summary(m_wls)$coefficients[2, 2],
      r2_ols       = summary(m_ols)$r.squared,
      r2_wls       = summary(m_wls)$r.squared,
      r2_loess     = r2_loess,
      n_bins       = n_bins,
      curvatura    = curvatura
    )
  }) %>%
  ungroup()

# C4. Bootstrap WLS para IC
boot_wls_fn <- function(data, indices) {
  d <- data[indices, ]
  if (nrow(d) < 2 || var(d$log_price_ratio) < 1e-10) return(NA_real_)
  coef(lm(log_q_med ~ log_price_ratio, data = d, weights = sqrt(d$n_obs)))[2]
}

ic_C <- agg_C %>%
  group_by(product_category) %>%
  group_modify(function(data, key) {
    if (nrow(data) < 4) return(tibble(ci_low_C=NA, ci_high_C=NA))
    b <- tryCatch(
      boot(data, boot_wls_fn, R = N_BOOT),
      error = function(e) NULL
    )
    if (is.null(b)) return(tibble(ci_low_C=NA, ci_high_C=NA))
    ci <- tryCatch(
      boot.ci(b, type = "perc", conf = 0.95)$percent[4:5],
      error = function(e) c(NA_real_, NA_real_)
    )
    tibble(ci_low_C = ci[1], ci_high_C = ci[2])
  }) %>%
  ungroup()

resultados_C_full <- left_join(resultados_C, ic_C, by = "product_category")

cat("\n  Elasticidades WLS con bins finos:\n")
print(resultados_C_full %>% arrange(elasticity_C) %>%
        mutate(across(c(elasticity_C, se_C, ci_low_C, ci_high_C,
                        r2_ols, r2_wls, r2_loess), round, 3)) %>%
        select(product_category, elasticity_C, se_C, ci_low_C, ci_high_C,
               r2_wls, r2_loess, n_bins, curvatura), n = 20)


cat(" DIAGNÓSTICO — ELASTICIDADES ANÓMALAS\n")


# Identificar categorías problemáticas en Ruta C
anomalas <- resultados_C_full %>%
  filter(is.na(elasticity_C) | elasticity_C > -0.05 | elasticity_C < -5) %>%
  pull(product_category)

if (length(anomalas) > 0) {
  cat(sprintf("\n  Categorías anómalas: %s\n", paste(anomalas, collapse = ", ")))
  cat("\n  Distribución de ventas por nivel de descuento:\n")

  diag_anomalas <- df %>%
    filter(product_category %in% anomalas, discount_applied > 0) %>%
    group_by(product_category, discount_applied) %>%
    summarise(
      n         = n(),
      med_units = median(units_sold),
      mean_units= round(mean(units_sold), 1),
      cv        = round(sd(units_sold) / mean(units_sold), 2),
      p25       = quantile(units_sold, 0.25),
      p75       = quantile(units_sold, 0.75),
      .groups   = "drop"
    )
  print(diag_anomalas, n = 40)
} else {
  cat("  ✓ Sin elasticidades anómalas en Ruta C\n")
}




cat(" MODELO GLOBAL CON ERRORES ROBUSTOS HC3\n")


modelo_global <- lm(log_q ~ log_price_ratio + product_category, data = df_valid)
ct_global     <- coeftest(modelo_global, vcov = vcovHC(modelo_global, "HC3"))
ci_global     <- coefci(modelo_global, vcov = vcovHC(modelo_global, "HC3"))

global_eps    <- ct_global["log_price_ratio", "Estimate"]
global_se     <- ct_global["log_price_ratio", "Std. Error"]

cat(sprintf("\n  Elasticidad global: %.3f (SE=%.3f)\n", global_eps, global_se))
cat(sprintf("  IC 95%%: [%.3f, %.3f]\n",
            ci_global["log_price_ratio", 1],
            ci_global["log_price_ratio", 2]))




cat(" SHRINKAGE BAYESIANO (ponderación por precisión)\n")


# Criterio de señal mejorado: KW + R²
señal_combinada <- resultados_C_full %>%
  left_join(kw_results %>% select(product_category, p_kw, n, señal_kw),
            by = "product_category") %>%
  mutate(
    # Señal fuerte: evidencia econométrica (R²) O estadística (KW) con suficiente n
    tiene_señal_v2 = (r2_wls >= 0.15) |
                     (señal_kw & !is.na(r2_wls) & r2_wls >= 0.05) |
                     (señal_kw & n >= 200),
    # Nivel de evidencia para calibrar la fuerza del shrinkage
    nivel_evidencia = case_when(
      r2_wls >= 0.70 & señal_kw                ~ "muy_alta",
      r2_wls >= 0.30 & señal_kw                ~ "alta",
      (r2_wls >= 0.10 | señal_kw) & n >= 100   ~ "media",
      TRUE                                       ~ "baja"
    )
  )

# Prior global ponderado por precisión
eps_raw_vec <- ifelse(
  is.na(señal_combinada$elasticity_C) | !señal_combinada$tiene_señal_v2,
  GLOBAL_ELASTICITY,
  señal_combinada$elasticity_C
)
se_vec <- ifelse(
  is.na(señal_combinada$se_C) | señal_combinada$se_C < 1e-6,
  1.5,
  señal_combinada$se_C
)

# Prior global como media ponderada por 1/SE² de las categorías con señal
mask_señal <- señal_combinada$tiene_señal_v2 & !is.na(señal_combinada$elasticity_C)
if (sum(mask_señal) >= 2) {
  pesos_prior <- 1 / se_vec[mask_señal]^2
  global_avg  <- weighted.mean(eps_raw_vec[mask_señal], pesos_prior)
} else {
  global_avg <- GLOBAL_ELASTICITY
}
cat(sprintf("\n  Prior global recalculado (ponderado por precisión): %.3f\n", global_avg))

# Fuerza del shrinkage según nivel de evidencia
shrink_strength <- case_when(
  señal_combinada$nivel_evidencia == "muy_alta" ~ SHRINKAGE_STRONG,
  señal_combinada$nivel_evidencia == "alta"     ~ 60,
  señal_combinada$nivel_evidencia == "media"    ~ 100,
  TRUE                                           ~ SHRINKAGE_BASE
)

# SE del prior global (estimado como SE de la media ponderada)
se_global <- global_se  # del modelo HC3

# Shrinkage ponderado por precisión:
# eps_shrunk = (eps_raw / se² + global / se_global²) / (1/se² + 1/se_global²)
# pero con penalización adicional según nivel de evidencia
precision_cat    <- 1 / se_vec^2
precision_global <- 1 / se_global^2 * (SHRINKAGE_BASE / shrink_strength)

eps_shrunk <- (precision_cat * eps_raw_vec + precision_global * global_avg) /
              (precision_cat + precision_global)

# Clamp a rango económicamente plausible
eps_shrunk <- pmax(pmin(eps_shrunk, -0.02), -5.0)

# IC shrinkage (proporcional al factor de contracción)
shrink_factor <- precision_cat / (precision_cat + precision_global)

resultados_shrink_v2 <- señal_combinada %>%
  mutate(
    eps_raw        = eps_raw_vec,
    eps_shrunk     = eps_shrunk,
    shrink_factor  = shrink_factor,
    ci_low_shrunk  = shrink_factor * ci_low_C  + (1 - shrink_factor) * global_avg,
    ci_high_shrunk = shrink_factor * ci_high_C + (1 - shrink_factor) * global_avg,
    nivel_evidencia = nivel_evidencia
  )

cat("\n  Elasticidades finales con shrinkage:\n")
print(
  resultados_shrink_v2 %>%
    arrange(eps_shrunk) %>%
    mutate(across(c(eps_shrunk, ci_low_shrunk, ci_high_shrunk,
                    shrink_factor, r2_wls), round, 3)) %>%
    select(product_category, eps_shrunk, ci_low_shrunk, ci_high_shrunk,
           shrink_factor, nivel_evidencia, r2_wls, curvatura),
  n = 20
)




cat(" COMPARACIÓN DE RUTAS A / B / C\n")


comparacion <- elasticidades_A %>%
  full_join(resultados_B %>% select(product_category, elasticity_B, r2_B),
            by = "product_category") %>%
  full_join(resultados_C_full %>% select(product_category, elasticity_C, r2_wls),
            by = "product_category") %>%
  full_join(resultados_shrink_v2 %>% select(product_category, eps_shrunk, nivel_evidencia),
            by = "product_category") %>%
  mutate(across(c(elasticity_A, elasticity_B, elasticity_C, eps_shrunk), round, 3),
         across(c(r2_B, r2_wls), round, 3))

cat("\n")
print(comparacion %>% arrange(eps_shrunk), n = 20)



theme_pricing <- theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", size = 12),
    plot.subtitle   = element_text(color = "grey40", size = 10),
    panel.grid.minor= element_blank(),
    legend.position = "bottom"
  )

color_map <- c(
  "muy_alta"  = "#2ecc71",
  "alta"      = "#27ae60",
  "media"     = "#e67e22",
  "baja"      = "#e74c3c"
)

# ── Plot 1: Forest plot — elasticidades finales ───────────────────────────────

p1 <- resultados_shrink_v2 %>%
  filter(!is.na(eps_shrunk)) %>%
  arrange(eps_shrunk) %>%
  mutate(product_category = factor(product_category, levels = product_category)) %>%
  ggplot(aes(x = eps_shrunk, y = product_category, color = nivel_evidencia)) +
  geom_vline(xintercept = -1,          linetype = "dashed", color = "grey60", linewidth = 0.5) +
  geom_vline(xintercept = global_avg,  linetype = "dotted", color = "steelblue", linewidth = 0.7) +
  geom_errorbarh(aes(xmin = ci_low_shrunk, xmax = ci_high_shrunk),
                 height = 0.3, linewidth = 0.8, alpha = 0.6) +
  geom_point(size = 3.5) +
  scale_color_manual(
    values = color_map,
    labels = c("muy_alta"="Muy alta", "alta"="Alta", "media"="Media", "baja"="Baja"),
    name   = "Evidencia"
  ) +
  labs(
    title    = "Elasticidades precio-demanda",
    subtitle = sprintf("Prior ponderado por precisión = %.2f | línea azul = prior global | línea gris = unitaria",
                       global_avg),
    x = "Elasticidad (shrunk)", y = NULL
  ) +
  theme_pricing

# ── Plot 2: Comparación rutas A / B / C ──────────────────────────────────────

p2 <- comparacion %>%
  filter(!is.na(eps_shrunk)) %>%
  pivot_longer(cols = c(elasticity_A, elasticity_B, elasticity_C, eps_shrunk),
               names_to = "ruta", values_to = "elasticity") %>%
  mutate(
    ruta = factor(ruta,
                  levels = c("elasticity_A","elasticity_B","elasticity_C","eps_shrunk"),
                  labels = c("A: OLS continuo","B: Jitter sim.","C: WLS bins","Shrunk final"))
  ) %>%
  filter(!is.na(elasticity)) %>%
  ggplot(aes(x = elasticity, y = reorder(product_category, elasticity),
             color = ruta, shape = ruta)) +
  geom_vline(xintercept = -1, linetype = "dashed", color = "grey60", linewidth = 0.4) +
  geom_point(size = 2.5, alpha = 0.8,
             position = position_dodge(width = 0.5)) +
  scale_color_manual(values = c("#3498db","#9b59b6","#e67e22","#2c3e50"),
                     name = "Método") +
  scale_shape_manual(values = c(16, 17, 15, 18), name = "Método") +
  labs(
    title    = "Convergencia entre rutas de estimación",
    subtitle = "Puntos cercanos = estimación robusta | divergencia = revisar datos",
    x = "Elasticidad estimada", y = NULL
  ) +
  theme_pricing

# ── Plot 3: Curvas de respuesta con bins finos ────────────────────────────────

cats_top <- resultados_shrink_v2 %>%
  filter(nivel_evidencia %in% c("muy_alta", "alta")) %>%
  pull(product_category)

p3 <- agg_C %>%
  filter(product_category %in% cats_top) %>%
  ggplot(aes(x = bin_midpoint, y = exp(log_q_med) - 1,
             color = product_category, size = n_obs)) +
  geom_ribbon(aes(ymin = exp(q25) - 1, ymax = exp(q75) - 1,
                  fill = product_category, group = product_category),
              alpha = 0.1, color = NA, linewidth = 0) +
  geom_line(linewidth = 0.8, aes(group = product_category)) +
  geom_point(alpha = 0.75) +
  geom_smooth(aes(weight = n_obs, group = product_category),
              method = "lm", formula = y ~ log(x),
              se = FALSE, linewidth = 0.4, linetype = "dashed", alpha = 0.5) +
  scale_size_continuous(range = c(1.5, 5), name = "n obs / bin") +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(breaks = c(1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Curvas de respuesta — categorías con señal alta",
    subtitle = "Mediana ± IQR | tamaño del punto = n obs por bin | línea discontinua = ajuste log-lineal",
    x = "Descuento (midpoint del bin)", y = "Unidades vendidas (mediana)",
    color = NULL, fill = NULL
  ) +
  theme_pricing +
  guides(size = guide_legend(order = 2), color = guide_legend(order = 1))

# ── Plot 4: R² comparativo (bins originales vs bins finos) ───────────────────

r2_comp <- resultados_C_full %>%
  select(product_category, r2_ols, r2_loess) %>%
  left_join(
    df_valid %>%
      group_by(product_category, discount_applied) %>%
      summarise(log_q_med = median(log_q),
                log_price_ratio = first(log(1 - discount_applied/100)),
                .groups="drop") %>%
      group_by(product_category) %>%
      summarise(r2_original = if(n() >= 2 && var(log_price_ratio) > 0)
                  summary(lm(log_q_med ~ log_price_ratio))$r.squared
                else NA_real_, .groups="drop"),
    by = "product_category"
  ) %>%
  pivot_longer(cols = c(r2_original, r2_ols, r2_loess),
               names_to = "metodo", values_to = "r2") %>%
  mutate(metodo = factor(metodo,
                         levels = c("r2_original","r2_ols","r2_loess"),
                         labels = c("4 puntos (original)","WLS bins finos","LOESS bins finos")))

p4 <- r2_comp %>%
  filter(!is.na(r2)) %>%
  ggplot(aes(x = reorder(product_category, r2, median),
             y = r2, fill = metodo)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_hline(yintercept = 0.15, linetype = "dashed", color = "red", linewidth = 0.5) +
  scale_fill_manual(values = c("#bdc3c7","#2980b9","#8e44ad"), name = NULL) +
  scale_y_continuous(labels = percent) +
  coord_flip() +
  labs(
    title    = "Mejora de R² con bins finos",
    subtitle = "Línea roja = umbral señal (R²=0.15)",
    x = NULL, y = "R²"
  ) +
  theme_pricing

# ── Plot 5: Factor de shrinkage por categoría ─────────────────────────────────
# Muestra cuánto se mueve cada categoría hacia el prior (0 = no se mueve, 1 = colapsa al prior)

p5 <- resultados_shrink_v2 %>%
  filter(!is.na(eps_shrunk)) %>%
  arrange(desc(shrink_factor)) %>%
  mutate(product_category = factor(product_category, levels = product_category)) %>%
  ggplot(aes(x = 1 - shrink_factor, y = product_category, fill = nivel_evidencia)) +
  geom_col(width = 0.7) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  scale_fill_manual(values = color_map,
                    labels = c("muy_alta"="Muy alta","alta"="Alta",
                               "media"="Media","baja"="Baja"),
                    name = "Evidencia") +
  scale_x_continuous(labels = percent, limits = c(0, 1)) +
  labs(
    title    = "Peso de los datos propios vs prior global",
    subtitle = "100% = estimación propia | 0% = colapsa al prior",
    x = "Peso de datos propios", y = NULL
  ) +
  theme_pricing

# ── Plot 6: Diagnóstico Wearables y otras anómalas ───────────────────────────

cats_diag <- if (length(anomalas) > 0) anomalas else
  resultados_shrink_v2 %>% filter(nivel_evidencia == "baja") %>%
  slice_head(n = 4) %>% pull(product_category)

p6 <- df %>%
  filter(product_category %in% cats_diag, discount_applied > 0) %>%
  ggplot(aes(x = factor(discount_applied), y = units_sold,
             fill = factor(discount_applied))) +
  geom_boxplot(outlier.size = 0.8, outlier.alpha = 0.4, linewidth = 0.4) +
  geom_jitter(width = 0.15, alpha = 0.2, size = 0.6) +
  facet_wrap(~product_category, scales = "free_y", nrow = 2) +
  scale_fill_brewer(palette = "Blues", guide = "none") +
  scale_x_discrete(labels = function(x) paste0(x, "%")) +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Diagnóstico: distribución de ventas por nivel de descuento",
    subtitle = "Categorías sin señal — verificar outliers y stock disponible",
    x = "Descuento aplicado", y = "Unidades vendidas"
  ) +
  theme_pricing

# ── Guardar todos los plots ────────────────────────────────────────────────────

ggsave(file.path(OUTPUT_DIR, "plot1_forest_elasticidades.png"),
       p1, width = 10, height = 7, dpi = 150)
ggsave(file.path(OUTPUT_DIR, "plot2_comparacion_rutas.png"),
       p2, width = 11, height = 7, dpi = 150)
ggsave(file.path(OUTPUT_DIR, "plot3_curvas_respuesta.png"),
       p3, width = 11, height = 6, dpi = 150)
ggsave(file.path(OUTPUT_DIR, "plot4_r2_mejora.png"),
       p4, width = 9,  height = 6, dpi = 150)
ggsave(file.path(OUTPUT_DIR, "plot5_shrinkage_factor.png"),
       p5, width = 9,  height = 6, dpi = 150)
ggsave(file.path(OUTPUT_DIR, "plot6_diagnostico_anomalas.png"),
       p6, width = 10, height = 7, dpi = 150)

cat("\n  Gráficos guardados en:", OUTPUT_DIR, "\n")


resultado_final <- comparacion %>%
  left_join(
    resultados_shrink_v2 %>%
      select(product_category, ci_low_shrunk, ci_high_shrunk,
             shrink_factor, nivel_evidencia, curvatura, tiene_señal_v2),
    by = "product_category"
  ) %>%
  left_join(kw_results %>% select(product_category, p_kw, n),
            by = "product_category") %>%
  mutate(
    nota = case_when(
      curvatura                      ~ "no-lineal: verificar GAM",
      !tiene_señal_v2                ~ "fallback al prior global",
      eps_shrunk > -0.1              ~ "elasticidad muy baja: revisar datos",
      TRUE                           ~ "ok"
    )
  )

write.csv(resultado_final,
          file.path(OUTPUT_DIR, "elasticidades_finales_v2.csv"),
          row.names = FALSE)


cat(" RESUMEN FINAL\n")

cat(sprintf("  Prior global recalculado   : %.3f\n", global_avg))
cat(sprintf("  Categorías con señal fuerte: %d / %d\n",
            sum(resultados_shrink_v2$nivel_evidencia %in% c("muy_alta","alta"), na.rm=TRUE),
            nrow(resultados_shrink_v2)))
cat(sprintf("  Categorías con curvatura   : %d\n",
            sum(resultados_shrink_v2$curvatura, na.rm=TRUE)))
cat(sprintf("  Categorías fallback al prior: %d\n",
            sum(!resultados_shrink_v2$tiene_señal_v2, na.rm=TRUE)))
cat(sprintf("  Rango elasticidades finales: [%.2f, %.2f]\n",
            min(resultados_shrink_v2$eps_shrunk, na.rm=TRUE),
            max(resultados_shrink_v2$eps_shrunk, na.rm=TRUE)))
cat(sprintf("  Resultados exportados a    : %s\n", OUTPUT_DIR))

cat("Análisis completo. gráficos +  CSV generados.\n")
