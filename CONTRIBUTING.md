# Contributing

Thanks for considering contributing to **mapext**! We welcome improvements of all kinds—bug fixes, documentation updates, new features, and more.

If you'd like to **report a bug**, **propose a feature**, or **submit a general issue**, please do so by opening an issue in the [GitHub Issues](https://github.com/tjrennie/mapext/issues) section of the repository.

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
  - [Reporting a new bug](#reporting-a-new-bug)
  - [Adding to an existing bug report](#adding-to-an-existing-bug-report)
- [Feature Requests](#feature-requests)
  - [Creating a new feature request](#creating-a-new-feature-request)
  - [Adding to an existing feature request](#adding-to-an-existing-feature-request)
- [Making Your Contribution](#making-your-contribution)
  - [Getting Started](#getting-started)
  - [Linting](#linting)
  - [Git Commit Scope](#git-commit-scope)
  - [Formatting Commit Messages](#formatting-commit-messages)
    - [Revert](#revert)
    - [Type](#type)
    - [Scope](#scope)
    - [Subject](#subject)
    - [Body](#body)
    - [Footer](#footer)
  - [Pulling Your Changes into the Main Branch](#pulling-your-changes-into-the-main-branch)

---

## Reporting Bugs

If you find a bug, please help us by submitting an issue. Before doing so, make sure:

- The bug hasn’t already been reported. Please search the existing issues to check if the issue has already been addressed.
- You’re using the latest version of the project (if applicable).

### Reporting a new bug

Please create an issue with the following:

- A clear and descriptive title.
- Steps to reproduce the issue.
- Expected vs. actual behavior.
- Screenshots or error logs (if relevant).
- Your environment (OS, browser, version, etc.).

### Adding to an existing bug report

- Please avoid opening a new issue. Instead, comment on the existing issue with any additional information you can provide.
- Make sure to check if the issue is still being worked on or if there are any updates.
- Adding context, such as your environment details or any new observations, can help the team resolve the issue more efficiently.

---

## Feature Requests

If you'd like to propose a new feature or improvement, please submit a feature request through an issue.

### Creating a new feature request

Please create an issue with the following:

- A clear and descriptive title.
- A detailed description of the feature.
- Why this feature would be beneficial or improve the project.
- Any potential use cases or examples of how this feature could be used.
- If applicable, any technical considerations or implementation ideas.

### Adding to an existing feature request

- Please avoid opening a new issue if a similar feature request already exists. Instead, comment on the existing issue with any additional information or thoughts you have.
- Check if the feature is actively being considered or planned.
- Adding context, such as examples, use cases, or technical ideas, can help the team assess and prioritize the feature more effectively.

---

## Making Your Contribution

Contributions must follow the project coding style (enforced via **Ruff** and **Black**) and maintain a clear, logical commit history. What follows is a guide on how to achieve this.

### Getting Started

1. **Set up your development environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Clone the repository:**

   ```bash
   git clone git@github.com:tjrennie/mapext.git
   cd mapext
   ```

3. **Install the package with development and documentation tools:**

   ```bash
   pip install -e .[dev,docs]  # On macOS: pip install -e ".[dev,docs]"
   ```

4. **Check for an existing branch** related to the changes you'd like to make. If one exists, switch to it:

   ```bash
   git checkout <existing-branch>
   ```

   If not, create a new branch:

   ```bash
   git checkout -b <your-initials>/<feature-name> main
   ```

### Linting

This project uses:

- [**Black**](https://black.readthedocs.io/en/stable/) for code formatting
- [**Ruff**](https://docs.astral.sh/ruff/) for linting

Before committing, run:

```bash
black .
ruff check .
```

You can also auto-fix issues with Ruff:

```bash
ruff check . --fix
```

### Git Commit Scope

Commit your changes in small, focused chunks. Try to **commit often**, clean up later, and publish once. Use descriptive messages that follow the formatting guide below.

```bash
git add -p
git commit -v
```

### Formatting Commit Messages

We follow the [Angular commit convention](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit):

```none
<type>(<scope>): <subject>

<body>

<footer>
```

**Header** is required. **Scope**, **Body**, and **Footer** are optional.

**Line limit:** Each line should be ≤ 100 characters.

Examples:

```bash
docs(changelog): update changelog to beta.5
```

```bash
fix(release): need to depend on latest rxjs and zone.js

The version in our package.json gets copied to the one we publish, and users need the latest of these.
```

#### Revert

If reverting a previous commit, start with `revert:` and include:

```none
This reverts commit <hash>.
```

#### Type

- **build**: Build system or dependency changes
- **docs**: Documentation only
- **feat**: New features
- **fix**: Bug fixes
- **perf**: Performance improvements
- **refactor**: Code refactoring
- **style**: Code style changes (formatting, etc.)
- **test**: Adding or updating tests

#### Scope

Optional. Typically the module or feature affected.

#### Subject

- Use **imperative present tense** (e.g., "add", not "added")
- Do **not** capitalize the first letter
- Do **not** end with a period

#### Body

Explain the **why** of the change. Use imperative tense. Include lists with `-` if needed.

#### Footer

Use for:

- **Breaking Changes**:

  ```none
  BREAKING CHANGE: changes method signature of foo.bar()
  ```

- **Referencing issues**:

  ```none
  Closes #234, #567
  ```

### Pulling Your Changes into the Main Branch

Before opening a pull request:

- Ensure your PR is clear, focused, and well-described
- Reference related issues
- Include or describe any new tests
- Rebase or merge `main` to ensure up-to-date changes
- Pass all CI checks
- Ensure your code is commented, documented, and linted

When you receive feedback, respond constructively and make necessary updates. CI checks must still pass after revisions.

---

Thanks again for your contributions to **mapext**!
