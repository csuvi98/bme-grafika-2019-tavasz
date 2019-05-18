// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================


//FELADAT

/*
Készítsen virtuális kaleidoszkópot. A tükörrendszer szabályos sokszög, amelynek az oldalszáma 3-ról indul és az ‘a’ billentyûvel lehet inkrementálni.
A tükör anyaga arany (‘g’) vagy ezüst (‘s’). 
A kaleidoszkóp végén legalább három, különbözõ anyagtulajdonságú ellipszoid található, amelyet ambiens+diffúz+Phong-Blinn spekuláris modell szerint veri vissza a fényt. 
Az ellipszoidokat véletlen irányú erõk érik (Brown mozgás) amit követve mozog.

A sima anyagok törésmutatója és kioltási tényezõje az R,G,B hullámhosszokon:

Arany (n/k): 0.17/3.1, 0.35/2.7, 1.5/1.9

Ezüst (n/k) 0.14/4.1, 0.16/2.3, 0.13/3.1
*/

#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	bool rough = true;
	bool reflective = false;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

inline vec3 operator/(const vec3& v, const vec3& u) { return vec3(v.x / u.x, v.y / u.y, v.z / u.z); }
const int maxdepth = 10;
static int mirrors = 3;

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir, reflectionDir;
	bool out = true;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
	Ray(vec3 _start, vec3 _dir, bool _out) { start = _start; dir = normalize(_dir); out = _out; }
};

class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};



class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / windowWidth - 1) + up * (2.0 * (Y + 0.5) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

class Ellipsoid : public Intersectable {

		vec3 center;
		vec3 radius;
public:
		Ellipsoid(const vec3& _center, vec3 _radius, Material* _material) {
			center = _center;
			radius = _radius;
			material = _material;
		}
		Hit intersect(const Ray& ray) {
			Hit hit;
			vec3 dist = ray.start - center;
			float a = (ray.dir.x * ray.dir.x / (radius.x*radius.x)) + (ray.dir.y* ray.dir.y / (radius.y*radius.y)) + (ray.dir.z* ray.dir.z / (radius.z*radius.z));
			float b = ((2.0*ray.dir.x * ray.start.x - 2.0*center.x * ray.dir.x) / (radius.x * radius.x)) +
				((2.0*ray.dir.y * ray.start.y - 2.0*center.y * ray.dir.y) / (radius.y * radius.y)) +
				((2.0*ray.dir.z * ray.start.z - 2.0*center.z * ray.dir.z) / (radius.z * radius.z));
			float c = ((center.x*center.x - 2.0*center.x*ray.start.x + ray.start.x*ray.start.x) / (radius.x*radius.x)) +
				((center.y*center.y - 2.0*center.y*ray.start.y + ray.start.y*ray.start.y) / (radius.y*radius.y)) +
				((center.z*center.z - 2.0*center.z*ray.start.z + ray.start.z*ray.start.z) / (radius.z*radius.z)) - 1.0f;
			float discr = b * b - 4.0 * a * c;
			if (discr < 0) return hit;
			float sqrt_discr = sqrtf(discr);
			float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
			float t2 = (-b - sqrt_discr) / 2.0 / a;
			if (t1 <= 0) return hit;
			hit.t = (t2 > 0) ? t2 : t1;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = vec3((hit.position.x - center.x) / (radius.x), (hit.position.y - center.y) / (radius.y), (hit.position.z - center.z) / (radius.z));
			hit.material = material;
			return hit;
		}

		
};




class Mirror : public Intersectable {
	vec3 pos;
	vec3 normal = vec3(1, 0, 0);
public:
	Mirror(vec3 _pos, vec3 _normal, Material* _material) {
		pos = _pos;
		normal = _normal;
		material = _material;
		
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float d = dot(normal, ray.dir);
		if (d == 0)
			return hit;
		hit.material = material;
		hit.normal = normal;

		float x = dot(pos - ray.start, normal) / d;
		if (x < 0) {
			return hit;
		}

		vec3 intersection = ray.dir * x + ray.start;

		hit.position = intersection;

		float t = length(intersection - ray.start);

		hit.t = t;

		return hit;

	}

	
};



class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
	
public:
	void build() {
		vec3 eye = vec3(0, 0, 7), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(0, 0, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kdr(0.3f, 0.2f, 0.1f);
		vec3 kd1(0.7f, 0.2f, 0.1f);
		vec3 kd2(0.0f, 0.5f, 0.7f);
		vec3 kd3(0.1f, 0.6f, 0.3f);
		vec3 kdw(0.0f, 1.0f, 1.0f);
		vec3 ks(2, 2, 2);
		Material * wmaterial = new Material(kdw, ks, 50);
		Material * material = new Material(kd1, ks, 50);
		Material * material2 = new Material(kd2, ks, 50);
		Material * material3 = new Material(kd3, ks, 50);
		Material * rmaterial = new Material(kdr, ks, 50);
		rmaterial->reflective = true;
		rmaterial->rough = false;
		objects.push_back(new Ellipsoid(vec3(0.1f, -0.1f, -1.0f), vec3(0.6, 0.3, 0.2), material2));
		objects.push_back(new Ellipsoid(vec3(0.0f, 0.0f, 0.0f),  vec3(0.4, 0.2, 0.3), material));
		objects.push_back(new Ellipsoid(vec3(-0.3f, 0.3f, 1.0f),  vec3(0.3, 0.2, 0.1), material3));

		float rad = 2 * M_PI / mirrors;
		for(int i = 0; i< mirrors; i++){
			float deg = i * rad - M_PI / 2;
			vec3 dir(cos(deg), sin(deg), 0);
			objects.push_back(new Mirror(dir*0.8f, dir*-1.0f, rmaterial));
		}
		
		
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 reflect(vec3 inDir, vec3 normal) {
		return inDir - normal * dot(normal, inDir) * 2.0f;
	}

	vec3 Fresnel(vec3 inDir, vec3 normal) {
		vec3 n(0.17f, 0.35f, 1.5f);
		vec3 kappa(3.1f, 2.7f, 1.9f);
		float cosa = -dot(inDir, normal);
		vec3 one(1, 1, 1);
		vec3 F0 = ((n - one)*(n - one) + kappa * kappa) / ((n + one)*(n + one) + kappa * kappa);

		return F0 + (one - F0) * pow(1 - cosa, 5);
	}

	

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > maxdepth) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = hit.material->ka * La;
		if (hit.material->rough) {
			for (Light * light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		if (hit.material->reflective) {
			
			vec3 reflectionDir = reflect(ray.dir, hit.normal);
			Ray reflectRay(hit.position + hit.normal * epsilon, reflectionDir, ray.out);
			outRadiance = outRadiance + trace(reflectRay, depth + 1)*Fresnel(ray.dir, hit.normal);
		}
		return outRadiance;
	}

	void resetMirrors() {
		for (int i = 0; i < mirrors; i++) {
			objects.pop_back();
		}
	}
};



GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;


// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture * pTexture;
public:
	void Create(std::vector<vec4>& image) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		pTexture = new Texture(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
	fullScreenTexturedQuad.Create(image);

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a') {
		scene.resetMirrors();
		mirrors++;
		glViewport(0, 0, windowWidth, windowHeight);
		scene.build();

		std::vector<vec4> image(windowWidth * windowHeight);
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		scene.render(image);
		long timeEnd = glutGet(GLUT_ELAPSED_TIME);
		printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
		fullScreenTexturedQuad.Create(image);
		
	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	float t = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	if ((int)t % 1 == 0) {
		//scene.animObj();
		//printf("\n plez");
	}
	glutPostRedisplay();
}