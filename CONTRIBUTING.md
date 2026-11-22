# Commit privacy

Enable the repository hooks after cloning and use the noreply email shown in your GitHub email settings. Set your Git name to your public contributor name.

```sh
git config --local core.hooksPath .githooks
git config --local user.name "PUBLIC-ALIAS"
git config --local user.email "YOUR-GITHUB-NOREPLY-ADDRESS"
```

The hooks require Python 3.8 or later. They check the Git author and committer, the proposed message, and newly introduced private values in staged files. Notebook code, textual outputs, and metadata are included. Word, PowerPoint, and Excel files are checked for personal creator and last-editor fields in their Office core metadata. Use a public alias or project/team attribution for those fields. Changes to a tracked file are compared with its previous version so editing an unrelated line does not require rewriting historical research evidence.

For private project names, handles, or other identifiers specific to your environment, put one literal per line in the local file returned by `git rev-parse --git-path info/privacy-identifiers`. The checker reports the affected file and category without printing the matched content.

Before committing, review document bodies, images, archives, and embedded media separately. Office inspection is limited to 100 MiB packages and 256 KiB metadata parts. Local hooks apply to commits made in this checkout. Web edits and other clones have their own settings.
