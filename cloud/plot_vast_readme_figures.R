#!/usr/bin/env Rscript
# Generate documentation-ready Vast.ai benchmark and actual-run figures.

suppressPackageStartupMessages(library(ggplot2))

root <- normalizePath(getwd(), mustWork = TRUE)
out_dir <- file.path(root, "cloud", "vast_cost_performance")
bench_csv <- file.path(out_dir, "vast_doc_benchmark_gpu_samples_data.csv")
actual_csv <- file.path(out_dir, "vast_doc_actual_sph_vs_dph_data.csv")
plot_date <- "2026-05-29"

pretty_gpu <- function(x) {
  x <- gsub("_", " ", x)
  x <- gsub("RTX PRO 6000 WS", "RTX Pro 6000 WS", x, fixed = TRUE)
  x <- gsub("A100 SXM4", "A100 SXM4", x, fixed = TRUE)
  x <- gsub("H100 NVL", "H100 NVL", x, fixed = TRUE)
  x <- gsub("RTX 5060 Ti", "RTX 5060 Ti", x, fixed = TRUE)
  x
}

palette <- c(
  "RTX 3090" = "#2B6CB0",
  "RTX 4090" = "#E67E22",
  "RTX 5060" = "#2F855A",
  "RTX 5060 Ti" = "#2C7A7B",
  "RTX 5070" = "#805AD5",
  "RTX 5090" = "#C53030",
  "RTX Pro 6000 WS" = "#795548",
  "A100 SXM4" = "#B83280",
  "H100 NVL" = "#4A5568"
)

base_theme <- function(base_size = 15) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 5, margin = margin(b = 6)),
      plot.subtitle = element_text(color = "#4A5568", size = base_size + 1, margin = margin(b = 12)),
      axis.title = element_text(face = "bold", size = base_size + 1),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "#E2E8F0"),
      panel.grid.major.x = element_line(color = "#EDF2F7"),
      strip.text = element_text(face = "bold", size = base_size + 2),
      legend.position = "bottom",
      legend.box = "vertical",
      legend.title = element_text(face = "bold", size = base_size),
      legend.text = element_text(size = base_size - 1),
      axis.text = element_text(size = base_size - 1),
      plot.caption = element_text(color = "#4A5568", hjust = 0, size = base_size - 2)
    )
}

# Benchmark figure -----------------------------------------------------------
bench <- read.csv(bench_csv, stringsAsFactors = FALSE)
bench$gpu_label <- pretty_gpu(bench$gpu)
bench$sample_label <- paste0(bench$n_samples, " samples")

order_df <- aggregate(surfaces_per_hour ~ gpu_label, data = subset(bench, n_samples == 20), median, na.rm = TRUE)
order_df <- order_df[order(-order_df$surfaces_per_hour), ]
bench$gpu_label <- factor(bench$gpu_label, levels = order_df$gpu_label)
bench$sample_label <- factor(bench$sample_label, levels = c("20 samples", "100 samples"))

bench_long <- rbind(
  data.frame(gpu_label = bench$gpu_label, sample_label = bench$sample_label,
             metric = "Throughput", value = bench$surfaces_per_hour),
  data.frame(gpu_label = bench$gpu_label, sample_label = bench$sample_label,
             metric = "Cost", value = bench$cost_per_1000_surfaces)
)
bench_long$metric <- factor(bench_long$metric, levels = c("Throughput", "Cost"),
                            labels = c("Throughput (surfaces/hour)", "Cost (USD per 1,000 surfaces)"))

pd <- position_dodge(width = 0.55)
benchmark_plot <- ggplot(bench_long, aes(x = value, y = gpu_label, color = sample_label)) +
  geom_point(position = pd, alpha = 0.28, size = 2.1) +
  stat_summary(fun = median, geom = "point", position = pd, size = 4.2, shape = 18) +
  facet_wrap(~ metric, scales = "free_x", nrow = 1) +
  scale_color_manual(values = c("20 samples" = "#2B6CB0", "100 samples" = "#E53E3E"), name = "Simulation size") +
  labs(
    title = paste0("Vast GPU benchmark results (", plot_date, ")"),
    subtitle = "Controlled benchmark runs of the surface-computation task; diamonds show median performance and faint points show individual runs.",
    x = NULL,
    y = NULL,
    caption = "Source: benchmark runs on Vast.ai GPU instances. Prices and availability change over time."
  ) +
  base_theme(15) +
  theme(
    legend.position = "bottom",
    axis.text.y = element_text(size = 13),
    panel.spacing.x = unit(24, "pt")
  )

ggsave(file.path(out_dir, "vast_doc_benchmark_gpu_samples.png"), benchmark_plot,
       width = 13.5, height = 7.2, dpi = 180, bg = "white")

# Actual-run figure ----------------------------------------------------------
actual <- read.csv(actual_csv, stringsAsFactors = FALSE)
actual <- subset(actual, gpu != "unknown" & is.finite(real_sph) & is.finite(price_per_hour))
actual$gpu_label <- pretty_gpu(actual$gpu)
actual$gpu_label <- factor(actual$gpu_label, levels = names(palette)[names(palette) %in% unique(actual$gpu_label)])

actual_plot <- ggplot(actual, aes(x = price_per_hour, y = real_sph)) +
  geom_point(aes(color = gpu_label, size = surface_count, alpha = span_hours), stroke = 0.35) +
  scale_color_manual(values = palette, name = "GPU", drop = TRUE) +
  scale_size_continuous(name = "Amount of measured work", range = c(3.2, 11), breaks = c(500, 1000, 2000, 2600), labels = c("500 surfaces", "1,000", "2,000", "2,600")) +
  scale_alpha_continuous(name = "Estimate precision", range = c(0.35, 0.95), breaks = c(0.5, 1.5, 2.5, 3.5), labels = c("low", "medium", "high", "highest")) +
  guides(
    color = guide_legend(order = 1, override.aes = list(size = 5, alpha = 0.9)),
    size = guide_legend(order = 2, override.aes = list(alpha = 0.65, color = "#4A5568")),
    alpha = guide_legend(order = 3, override.aes = list(size = 6, color = "#4A5568"))
  ) +
  labs(
    title = paste0("Actual Vast run: throughput vs hourly price (", plot_date, ")"),
    subtitle = "Large-scale 20-sample run on Vast.ai. Larger points use more completed surfaces; darker points are more precise estimates because the machine worked longer.",
    x = "Vast price (USD/hour)",
    y = "Actual throughput (surfaces/hour)",
    caption = "Source: completed surface batches from the production run and Vast.ai price metadata. Prices and availability change over time."
  ) +
  base_theme(15) +
  theme(
    legend.position = "right",
    legend.box = "vertical",
    legend.key.height = unit(18, "pt"),
    plot.margin = margin(10, 14, 10, 10)
  )

ggsave(file.path(out_dir, "vast_doc_actual_sph_vs_dph.png"), actual_plot,
       width = 12.5, height = 7.4, dpi = 180, bg = "white")

cat("wrote", file.path(out_dir, "vast_doc_benchmark_gpu_samples.png"), "\n")
cat("wrote", file.path(out_dir, "vast_doc_actual_sph_vs_dph.png"), "\n")
