---
layout: page
title: "DeepDados - Blog"
subtitle: Projetos - Inteligência Artificial 
css: "/css/index.css"
meta-title: "Projetos - AI (Autores: César Pedrosa Soares e Lucas Pedrosa Soares"
meta-description: "Projetos - AI (Autores: César Pedrosa Soares e Lucas Pedrosa Soares"
---

<div class="list-filters">
  <span class="list-filter filter-selected">Projeto COVID-19 e Inteligência Artificial </span>
</div>

<div class="posts-list">
  {% for post in site.tags.rms %}
  <article>
    <a class="post-preview" href="{{ post.url | prepend: site.baseurl }}">
	    <h2 class="post-title">{{ post.title }}</h2>
	
	    {% if post.subtitle %}
	    <h3 class="post-subtitle">
	      {{ post.subtitle }}
	    </h3>
	    {% endif %}
      <p class="post-meta">
        Posted on {{ post.date | date: "%B %-d, %Y" }}
      </p>

      <div class="post-entry">
        {{ post.content | truncatewords: 50 | strip_html | xml_escape}}
        <span href="{{ post.url | prepend: site.baseurl }}" class="post-read-more">[Read&nbsp;More]</span>
      </div>
    </a>  
   </article>
  {% endfor %}
</div>
