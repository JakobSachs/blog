package main

import (
	bf "github.com/russross/blackfriday/v2"

	"bufio"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type Post struct {
	Title   string
	Type    string
	Date    time.Time
	DateStr string
	Content string
	HTML    template.HTML
}

// Returns the list of all markdown files in the given directory. Returns errors if any occur
func GetPostFiles(contentDir string) ([]string, error) {
	var files []string
	err := filepath.Walk(contentDir, func(path string, info fs.FileInfo, err error) error {
		if err != nil { // handle walking error
			return err
		}

		if !info.IsDir() && strings.HasSuffix(info.Name(), ".md") {
			files = append(files, path)
		}
		return nil
	})

	return files, err
}

// Parses out a Post struct from the given markdown file
// func PostFromFile(path string) (Post, error) {
// 	var post Post
// 	// read in file line-by-line
// 	fs, err := os.Open(path)
// 	if err != nil {
// 		return post, err
// 	}
// 	defer fs.Close()

// 	var lines []string
// 	scanner := bufio.NewScanner(fs)
// 	for scanner.Scan() {
// 		lines = append(lines, scanner.Text())
// 	}
// 	if err := scanner.Err(); err != nil {
// 		return post, err
// 	}
// 	if len(lines) < 2 {
// 		return post, errors.New("file has too few lines! Cant have a shorter then 2 lines post")
// 	}

// 	// parse out the lines
// 	titleLine := lines[0]
// 	sections := strings.Split(titleLine, "-")

// 	if len(sections) != 3 {
// 		return post, errors.New("first line has malformed header info, needs to be ")
// 	}

// 	post.Date, err = time.Parse("01.02.2006", strings.TrimSpace(sections[0][1:])) // [1:] to remove the leading '#'
// 	if err != nil {                                                               // cant parse date
// 		return post, err
// 	}
// 	post.DateStr = post.Date.Format("01.02.2006")
// 	post.Type = strings.TrimSpace(sections[1])
// 	post.Title = strings.TrimSpace(sections[2])
// 	post.Content = strings.Join(lines[1:], "") // simply just join rest of content for now

// 	return post, nil
// }

// Parses out a Post struct from the given markdown file
func PostFromFile(path string) (Post, error) {
	var post Post
	// read in file line-by-line
	fs, err := os.Open(path)
	if err != nil {
		return post, err
	}
	defer fs.Close()

	var lines []string
	scanner := bufio.NewScanner(fs)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return post, err
	}
	if len(lines) < 2 {
		return post, errors.New("file has too few lines! Cant have a shorter then 2 lines post")
	}

	// parse out the lines
	titleLine := lines[0]
	sections := strings.Split(titleLine, "-")

	if len(sections) != 3 {
		return post, errors.New("first line has malformed header info, needs to be ")
	}

	post.Date, err = time.Parse("02.01.2006", strings.TrimSpace(sections[0][1:])) // [1:] to remove the leading '#'
	if err != nil {                                                               // cant parse date
		return post, err
	}
	post.DateStr = post.Date.Format("02.01.2006")
	post.Type = strings.TrimSpace(sections[1])
	post.Title = strings.TrimSpace(sections[2])

	// join the markdown content, starting after the title
	post.Content = strings.Join(lines[1:], "\n")

	// convert markdown to HTML using Blackfriday
	post.HTML = template.HTML(bf.Run([]byte(post.Content)))

	return post, nil
}

// populates the home page (recent postings feed) based on our list of posts
func RenderHomePage(posts []Post) error {
	tmpl, err := template.ParseFiles("templates/home.html")
	if err != nil {
		return err
	}

	out_file, err := os.Create("output/home.html")
	if err != nil {
		return err
	}
	defer out_file.Close()

	// render & write output to file
	if err = tmpl.Execute(io.Writer(out_file), posts); err != nil {
		return err
	}

	return nil
}

func RenderPost(post Post) error {
	tmpl, err := template.ParseFiles("templates/post.html")
	if err != nil {
		return err
	}

	out_file, err := os.Create(fmt.Sprintf("output/%s-%s.html", post.DateStr, post.Title))
	if err != nil {
		return err
	}
	defer out_file.Close()

	// render & write output to file
	if err = tmpl.Execute(io.Writer(out_file), post); err != nil {
		return err
	}

	return nil
}

// generates the statically generated site
func main() {
	fmt.Fprintf(os.Stderr, "Generating site...\n")

	// load posts
	files, err := GetPostFiles("content")
	if err != nil {
		fmt.Fprintln(os.Stderr, "unable to find post files with err: ", err)
		os.Exit(1)
	}

	var posts []Post
	for _, f := range files {
		p, err := PostFromFile(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, "unable to parse post: ", f, "with err: ", err)
			os.Exit(1)
		}
		posts = append(posts, p)
	}

	// sort posts by date
	sort.Slice(posts, func(i, j int) bool {
		return posts[i].Date.After(posts[j].Date)
	})

	for _, p := range posts {
		err := RenderPost(p)
		if err != nil {
			fmt.Fprintln(os.Stderr, "unable to render post: ", p, "with err: ", err)
			os.Exit(1)
		}
	}

	if err := RenderHomePage(posts); err != nil {
		fmt.Fprint(os.Stderr, "unable to render home-page with err: ", err)
		os.Exit(1)
	}

	// copy over other files
	cmd := exec.Command("sh", "-c", "cp templates/*.css output/")

	if err := cmd.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "Unable to copy over sylesheets: ", err.Error())
		os.Exit(1)
	}

}
