(ns personal-site.core
  (:require [markdown.core :as md]
            [selmer.parser :as selmer]
            [clojure.java.io :as io]
            [me.raynes.fs :as fs]
            [toml-clj.core :as toml])
  (:gen-class))

(defn parse-front-matter [content]
  (let [[_ fm body] (re-matches
                     #"(?s)^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n(.*)$"
                     content)]
    (if fm
      {:meta (toml/read-string fm {:key-fn keyword})
       :body body}
      {:meta {}
       :body content})))


(defn render-post [md-file]
  (let [content (slurp md-file)
        {:keys [meta body]} (parse-front-matter content)
        html (md/md-to-html-string body)]
    {:meta meta
     :body html}))

(defn save-html [filename html]
  (spit filename html))

(defn copy-assets [src-dir dest-dir]
  ;; Copy all non-markdown files from src-dir to dest-dir
  (doseq [f (fs/list-dir src-dir)]
    (when-not (.endsWith (.getName f) ".md")
      (fs/copy+ f (fs/file dest-dir (.getName f))))))

(defn process-post [p]
  (if (.isDirectory p)
    ;; Folder-style post
    (let [md-file (first (filter #(-> % .getName (.endsWith ".md"))
                                 (fs/list-dir p)))
          {:keys [meta body]} (render-post md-file)
          title (:title meta)
          out-dir (fs/file "public" (fs/base-name p))
          out-file (fs/file out-dir "index.html")]
      (when-not (:draft meta)
        (fs/mkdirs out-dir)
        (copy-assets p out-dir)
        (save-html out-file
                   (selmer/render-file "templates/post.html"
                                       {:title title
                                        :date (:date meta)
                                        :body body}))
        {:title title
         :date (:date meta)
         :file (str (fs/base-name p) "/index.html")}))

    ;; Flat markdown file
    (let [{:keys [meta body]} (render-post p)
          title (:title meta)
          out-file (fs/file "public" (str (fs/base-name p true) ".html"))]

      (when-not (:draft meta)
        (save-html out-file
                   (selmer/render-file "templates/post.html"
                                       {:title title
                                        :date (:date meta)
                                        :body body}))
        {:title title
         :date (:date meta)
         :file (.getName out-file)}))))

(defn -main [& _]
  ;; Ensure public/ exists
  (fs/mkdirs "public")

  ;; Process posts
  (let [post-paths (fs/list-dir "content/posts")
        posts (remove nil? (map process-post post-paths))]
    (save-html "public/index.html"
               (selmer/render-file "templates/index.html" {:posts posts}))
    (println "✅ Site generated in /public")))
