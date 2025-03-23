# Personal Notes and Learning Website

This repository contains my personal notes and learning materials, hosted using GitHub Pages with Jekyll. It includes various topics in machine learning, data structures, algorithms, and more. The site uses Jekyll for static site generation and Mermaid for diagram rendering.

## Repository Structure

```
.
├── index.md                 # Main landing page
├── notes/                   # Main content directory
│   ├── ml/                 # Machine Learning notes
│   ├── dsa/               # Data Structures & Algorithms
│   ├── reinforcement/     # Reinforcement Learning
│   ├── supervised/        # Supervised Learning
│   ├── unsupervised/     # Unsupervised Learning
│   ├── mlops/            # MLOps notes
│   └── bedrock_and_prompt_engineering/  # AWS Bedrock & Prompt Engineering
├── _includes/              # Jekyll includes directory
│   └── header.html        # Custom header override
├── _site/                 # Generated site (not tracked in git)
├── style.css              # Custom CSS styles
├── setup_jekyll.py        # Setup script for local development
├── Gemfile                # Ruby dependencies
└── Gemfile.lock           # Locked Ruby dependencies
```

## Prerequisites

- **Ruby** – Required for Jekyll (install via [Homebrew](https://brew.sh) or [ruby-lang.org](https://www.ruby-lang.org))
- **Bundler** – For managing Ruby dependencies
- **Python 3** – For running the setup script

## Getting Started

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Run the Setup Script**

   The Python script will set up your local environment:

   ```bash
   python3 setup_jekyll.py
   ```

   This script:
   - Verifies Ruby installation
   - Installs Bundler if needed
   - Sets up the Gemfile
   - Installs dependencies
   - Starts the Jekyll server at [http://localhost:4000](http://localhost:4000)

3. **View the Site**

   Once the server is running, visit [http://localhost:4000](http://localhost:4000) to view your local version of the site.

## Customization

- **Custom Header:**  
  The `_includes/header.html` provides a custom header implementation.

- **Content:**  
  Add or modify notes in the `notes/` directory. Each subdirectory represents a different topic area.

- **Styling:**  
  Custom styles can be added to `style.css`.

## Features

- Organized notes by topic
- Mermaid diagram support
- Custom styling
- Mobile-responsive design
- Easy local development setup

## Further Reading

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Mermaid.js Documentation](https://mermaid.js.org)