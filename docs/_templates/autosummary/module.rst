{{ fullname | escape | underline }}

{% if modules and "." in fullname %}
.. automodule:: {{ fullname }}
   :no-members:
{% else %}
.. automodule:: {{ fullname }}
{% endif %}

{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
