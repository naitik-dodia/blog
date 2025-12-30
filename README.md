# Personal Blog

This is my personal blog for sharing notes, theory, and practical knowledge.

## Getting Started

### Prerequisites

Ensure you have **Ruby** installed, then follow the official guides to install [Jekyll](https://jekyllrb.com/docs/installation/) and [Bundler](https://bundler.io/).

```bash
gem install jekyll bundler
bundle install
```

## Development

To run the site locally and see your changes in real-time:

```bash
bundle exec jekyll serve
```

The site will be available at `http://localhost:4000/blog/`.

### Local root serving
If you want to serve the site at the root (`http://localhost:4000/`) locally:

```bash
bundle exec jekyll serve --baseurl ""
```

## Build

To build the static site:

```bash
bundle exec jekyll build
```

The output will be in the `_site` directory.

---
Have fun.