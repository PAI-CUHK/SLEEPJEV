# SLEEPJEV release plots
#
# Source: final two-seed averages supplied for the release summary.
# Timing boundary: warm serving only; EDF I/O and overnight cache construction
# are excluded. These values should be replaced by a versioned experiment
# manifest before making an external benchmark claim.

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

output_dir <- file.path("docs", "assets", "plots")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

teal <- "#0B8F87"
slate <- "#667085"
ink <- "#182230"
grid <- "#E4E7EC"
muted <- "#667085"

high_q <- data.frame(
  Q = c(8, 32, 128, 512),
  sleepjev_hit5 = c(0.7153, 0.6532, 0.6735, 0.6704),
  independent_hit5 = c(0.6182, 0.6374, 0.6663, 0.6666),
  sleepjev_steady_ms = c(0.213, 0.245, 0.359, 0.833),
  independent_warm_ms = c(0.343, 0.726, 2.248, 8.983),
  sleepjev_qai = c(0.2934, 0.2757, 0.2731, 0.2683),
  independent_qai = c(0.3356, 0.2816, 0.2815, 0.2747)
)

high_q$serving_speedup <- high_q$independent_warm_ms / high_q$sleepjev_steady_ms
high_q$sleepjev_qa_qps <- high_q$Q * high_q$sleepjev_qai / high_q$sleepjev_steady_ms
high_q$independent_qa_qps <- high_q$Q * high_q$independent_qai / high_q$independent_warm_ms

base_theme <- theme_classic(base_family = "sans", base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15, colour = ink),
    plot.subtitle = element_text(size = 10.5, colour = muted, margin = margin(b = 10)),
    axis.title = element_text(face = "bold", colour = ink),
    axis.text = element_text(colour = ink),
    axis.line = element_line(colour = ink, linewidth = 0.45),
    panel.grid.major.y = element_line(colour = grid, linewidth = 0.35),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.title = element_blank(),
    legend.key.width = unit(1.5, "lines"),
    plot.margin = margin(10, 18, 10, 10)
  )

save_plot <- function(plot, name) {
  ggsave(file.path(output_dir, paste0(name, ".png")), plot, width = 8.6, height = 4.8,
         units = "in", dpi = 320, bg = "white")
  ggsave(file.path(output_dir, paste0(name, ".svg")), plot, width = 8.6, height = 4.8,
         units = "in", device = "svg", bg = "white")
  ggsave(file.path(output_dir, paste0(name, ".pdf")), plot, width = 8.6, height = 4.8,
         units = "in", device = "pdf", bg = "white")
}

hit5_long <- rbind(
  data.frame(Q = high_q$Q, method = "SLEEPJEV", value = high_q$sleepjev_hit5),
  data.frame(Q = high_q$Q, method = "Independent DL", value = high_q$independent_hit5)
)

p_hit5 <- ggplot(hit5_long, aes(Q, value, colour = method, group = method)) +
  geom_line(linewidth = 1.05) +
  geom_point(size = 2.8) +
  scale_colour_manual(values = c("SLEEPJEV" = teal, "Independent DL" = slate)) +
  scale_x_continuous(trans = "log2", breaks = high_q$Q, labels = comma) +
  scale_y_continuous(limits = c(0.58, 0.74), breaks = seq(0.60, 0.74, 0.02),
                     labels = label_number(accuracy = 0.02)) +
  labs(
    title = "Hit@5 remains stable as query load grows",
    subtitle = "High-query regime; two-seed mean; Q shown on a log2 axis",
    x = "Runtime query count (Q)", y = "Hit@5"
  ) +
  base_theme
save_plot(p_hit5, "high_q_hit5")

p_speed <- ggplot(high_q, aes(Q, serving_speedup)) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = muted, linewidth = 0.55) +
  geom_line(colour = teal, linewidth = 1.15) +
  geom_point(colour = teal, size = 3) +
  geom_text(aes(label = paste0(number(serving_speedup, accuracy = 0.01), "x")),
            vjust = -0.9, colour = teal, family = "sans", fontface = "bold", size = 3.6) +
  scale_x_continuous(trans = "log2", breaks = high_q$Q, labels = comma) +
  scale_y_continuous(limits = c(0, 12), breaks = seq(0, 12, 2),
                     labels = function(x) paste0(x, "x")) +
  labs(
    title = "Warm serving speedup widens with query count",
    subtitle = "Relative to Independent DL warm serving; SLEEPJEV cache and plan are reused",
    x = "Runtime query count (Q)", y = "Serving speedup"
  ) +
  base_theme +
  theme(legend.position = "none")
save_plot(p_speed, "serving_speedup")

qps_long <- rbind(
  data.frame(Q = high_q$Q, method = "SLEEPJEV", value = high_q$sleepjev_qa_qps),
  data.frame(Q = high_q$Q, method = "Independent DL", value = high_q$independent_qa_qps)
)

p_qps <- ggplot(qps_long, aes(Q, value, colour = method, group = method)) +
  geom_line(linewidth = 1.05) +
  geom_point(size = 2.8) +
  scale_colour_manual(values = c("SLEEPJEV" = teal, "Independent DL" = slate)) +
  scale_x_continuous(trans = "log2", breaks = high_q$Q, labels = comma) +
  scale_y_log10(labels = label_number(accuracy = 1), breaks = c(10, 20, 50, 100, 200)) +
  labs(
    title = "Quality-adjusted query throughput scales with reuse",
    subtitle = "QA-QPS = Q * sqrt(hit@5 * PU-recall@5) / warm serving time (ms)",
    x = "Runtime query count (Q)", y = "QA-QPS"
  ) +
  base_theme
save_plot(p_qps, "qa_qps")

write.csv(high_q, file.path(output_dir, "high_q_summary.csv"), row.names = FALSE)
