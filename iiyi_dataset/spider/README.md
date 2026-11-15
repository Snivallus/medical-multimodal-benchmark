# iiyi Case Spider

This folder contains the spider scripts used to crawl medical case data from [iiyi.com](https://bingli.iiyi.com/).

Everything runs on plain HTTP requests; no browser and no WebDriver are required.

---

## Folder Structure

```bash
spider/
├── README.md            # This file
├── .gitignore
├── crawl_case_list.py   # Collect case titles and URLs from the index pages
├── merge_cases.py       # Merge the collected lists into one
├── crawl_cases.py       # Crawl full case content and download images
└── merged_cases.json    # The merged case list (titles and URLs only)
```

Most generated files are ignored by `.gitignore`, except for `merged_cases.json`,
which records the merged case list used for crawling.

---

## Two Page Templates

The site publishes every case through two templates, and both are supported:

| Template | URL | Anchors |
| -------- | --- | ------- |
| desktop | `https://bingli.iiyi.com/show/{case_id}-1.html` | `.case_details`, `.case_summary`, `.case_study` |
| mobile | `https://m.iiyi.com/bl/d-{case_id}.html` | `.article-details`, `.abstract`, `.case-container` |

The desktop template is preferred: it carries the whole image gallery
(`.case_focus_map`) on top of the inline images, so it yields strictly more
images for the same case. The mobile template is the fallback, and it is the
only one that still uses the legacy anchors the old pipeline was written for.

`crawl_cases.py --site auto` follows the template of the input URL and switches
to the other one whenever a page cannot be parsed, so the crawl survives a
redesign of either side.

---

## Data Crawling Pipeline

The sub-workflow consists of three steps.

### 1. Collect case lists

Run `crawl_case_list.py` to collect case titles and URLs from the three feeds:

- [推荐案例](https://bingli.iiyi.com/) → `recommended_cases.json`
- [精选案例](https://bingli.iiyi.com/cull/) → `selected_cases.json`
- [最新案例](https://bingli.iiyi.com/news/) → `newest_cases.json`

The index pages load more cases by posting `p=<page>` to the same URL and
answering with a JSON payload that holds the next block of cases; the script
uses that endpoint directly. Only the 推荐案例 feed actually supports paged
loading, so the other two feeds contribute their first page.

Each file contains only case titles and URLs.

### 2. Merge case lists

Run `merge_cases.py` to merge the three case lists into `merged_cases.json`.
Cases are de-duplicated by URL, which also removes the overlap between feeds.
This file is the input for the next step.

### 3. Crawl full case content

Run `crawl_cases.py` with `merged_cases.json` as input to crawl full case content and download images.

```bash
Spider/
├── succeeded_cases.json    # Cases that have been successfully crawled
├── failed_cases.json       # Cases that failed to be crawled
├── raw_images/             # Images downloaded from the website
│   ├── case_0000/
|   |   ├── img_00.jpg
|   |   ├── img_01.jpg
|   |   └── ...
│   ├── case_0001/
|   |   ├── img_00.jpg
|   |   ├── img_01.jpg
|   |   └── ...
│   └── ...
└── ...
```

The step takes a few hours to complete, mostly because the crawler throttles
itself to avoid being banned.

Useful options:

```bash
python crawl_cases.py --limit 20          # smoke test on the first 20 cases
python crawl_cases.py --site mobile       # force the legacy mobile template
python crawl_cases.py --link-field alternate_url  # crawl the mobile URLs instead
python crawl_cases.py --no-resume         # ignore the checkpoint and start over
```

A crawl can be interrupted at any time. Every finished case is appended to the
`temp.jsonl` checkpoint as one JSON line, so a killed run never loses more than
the case being processed. The checkpoint is removed once `succeeded_cases.json`
has been written; the next run then resumes from that output file, so re-running
the same command tops the crawl up instead of starting over. Pass `--no-resume`
to crawl everything again from scratch.

---

## Notes

- Images are de-duplicated per case: the gallery repeats every picture as a
  large preview plus thumbnails, and its `_w150` previews collapse onto the
  full-size file, which is the one downloaded when it is still online.
- Site chrome (share icons, doctor avatars, QR codes, banners) is filtered out;
  only images served from `.../bingli/...` are kept as case media.
- Image files smaller than 5 KB are dropped as icons, and a case folder without
  any surviving image is removed, so "has images" always means "has files".
- `case_rewrite.py` and `case_filter.py` read the fields produced here
  (`标题`, `关键词`, `简要描述`, `摘要`, `内容`, `images`), so the schema is kept
  unchanged.
- A request burst makes the site answer with a small verification page instead
  of a case. The crawler retries once after a pause and then falls back to the
  other template; after a wall it also pauses 30-60 s so the rest of the run
  does not keep hitting it. A case that stays blocked is written to
  `failed_cases.json` and is retried by the next run.
- A few cases are login-gated on the desktop template but readable on the
  mobile one, which is one more reason the fallback exists.
