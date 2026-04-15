# SpamAssassin Integration

This directory holds the local benchmark-side setup for Apache SpamAssassin.

Official references:

- https://spamassassin.apache.org/doc.html
- https://svn.apache.org/repos/asf/spamassassin/trunk/INSTALL
- https://spamassassin.apache.org/full/4.0.x/doc/spamassassin-run.html
- https://spamassassin.apache.org/full/4.0.x/doc/sa-update.html

Files here:

- `50_local_benchmark.cf`: benchmark-oriented local config
- `user_prefs`: stable prefs file path used by the wrapper
- `install_spamassassin.sh`: install and lint helper
- `update_rules.sh`: rule refresh helper

Runtime wrapper:

- `../spamassassin.py`

The Python wrapper converts each CSV row into an RFC822 email message, pipes it to the
`spamassassin` CLI, and then parses `X-Spam-Status` / `X-Spam-Flag` from the output.
