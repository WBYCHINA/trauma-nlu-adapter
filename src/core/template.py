from jinja2 import Environment, FileSystemLoader
import os

class TemplateEngine:
    def __init__(self):
        template_dir = os.path.join(os.path.dirname(__file__), '../config/templates')
        os.makedirs(template_dir, exist_ok=True)
        
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False
        )
        
        self._create_default_templates()
    
    def _create_default_templates(self):
        injury_template = """你遇到的是【{{ injury_type }}】，位置在【{{ body_location }}】，严重程度为【{{ severity }}】。

急救步骤：
{% for i, step in enumerate(steps, 1) %}
{{ i }}. {{ step }}
{% endfor %}

注意事项：
{% for precaution in precautions %}
- {{ precaution }}
{% endfor %}

绝对不要做的事：
{% for contra in contraindications %}
- {{ contra }}
{% endfor %}
"""
        
        fsm_response_template = """{{ prompt }}"""
        
        combined_template = """{{ fsm_prompt }}

{% if injury_guide %}

根据你的描述，我判断伤情如下：
【{{ injury_guide.injury_type }}】- 【{{ injury_guide.body_location }}】- 【{{ injury_guide.severity }}】

具体处理步骤：
{% for i, step in enumerate(injury_guide.steps, 1) %}
{{ i }}. {{ step }}
{% endfor %}

重要提示：
{% for precaution in injury_guide.precautions %}
- {{ precaution }}
{% endfor %}
{% endif %}
"""
        
        templates_dir = os.path.join(os.path.dirname(__file__), '../config/templates')
        
        with open(os.path.join(templates_dir, 'injury_guide.j2'), 'w', encoding='utf-8') as f:
            f.write(injury_template)
        
        with open(os.path.join(templates_dir, 'fsm_response.j2'), 'w', encoding='utf-8') as f:
            f.write(fsm_response_template)
        
        with open(os.path.join(templates_dir, 'combined.j2'), 'w', encoding='utf-8') as f:
            f.write(combined_template)
    
    def render_injury_guide(self, injury_guide):
        template = self.env.get_template('injury_guide.j2')
        return template.render(
            injury_type=injury_guide['injury_type'],
            body_location=injury_guide['body_location'],
            severity=injury_guide['severity'],
            steps=injury_guide['steps'],
            precautions=injury_guide['precautions'],
            contraindications=injury_guide['contraindications']
        )
    
    def render_fsm_response(self, prompt):
        template = self.env.get_template('fsm_response.j2')
        return template.render(prompt=prompt)
    
    def render_combined(self, fsm_prompt, injury_guide=None):
        template = self.env.get_template('combined.j2')
        return template.render(fsm_prompt=fsm_prompt, injury_guide=injury_guide)
