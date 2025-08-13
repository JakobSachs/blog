(defproject bloggen "0.1.0-SNAPSHOT"
  :description "A simple static site generator"
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [markdown-clj "1.11.6"]       ;; Markdown → HTML
                 [hiccup "2.0.0-RC3"]           ;; HTML generation
                 [selmer "1.12.61"]
                 [me.raynes/fs "1.4.6"]
                 [io.github.tonsky/toml-clj "0.1.0"]
                 [clj-commons/fs "1.6.310"]]    ;; File utils

  :main personal-site.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
