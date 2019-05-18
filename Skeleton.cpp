
//=============================================================================================

//
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
Írjon 2D egykerekû bicikli szimulátor programot. 
A pálya negatív tenziójú explicit(!) Kochanek-Bartels spline, amelynek kontrollpontjait az egér bal gombjának lenyomásával lehet megadni, 
tetszõleges idõpontban és sorrendben. A bicikli vonalas ábra, forgó, többküllõs kerékkel és vadul pedálozó biciklissel. 
A kamera ortogonális vetítéssel dolgozik, kezdetben rögzített, majd SPACE hatására a biciklizõt követi. 
A biciklis a kezdeti kamera két széle között teker fel és alá, amit kb. 5 másodperc alatt tesz meg. 
A biciklis állandó erõvel teker, amivel a nehézségi erõt és a sebességgel arányos légellenállást küzdi le. 
A tömeg az izmok erejéhez képest elhanyagolható, így a sebesség gyorsan változhat.

A távoli háttér egy procedurálisan textúrázott téglalap, ami az égboltot és pozitív tenziójú explicit(!) Kochanek-Bartels spline-nal definiált hegységet ábrázol.
A távoli háttér nem követi a kamerát a hamis perspektíva érdekében.
*/


#include "framework.h"
// vertex shader in GLSL
const char * vertexSource = R"(
    #version 330
    precision highp float;
 
    uniform mat4 MVP;            // Model-View-Projection matrix in row-major format
 
    layout(location = 0) in vec2 vertexPosition;    // Attrib Array 0
 
    void main() {
        gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;         // transform to clipping space
    }
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
    #version 330
    precision highp float;
 
    uniform vec3 color;
    out vec4 fragmentColor;        // output that goes to the raster memory as told by glBindFragDataLocation
 
    void main() {
        fragmentColor = vec4(color, 1); // extend RGB to RGBA
    }
)";
bool track = false;

class Camera2D {
	vec2 wCenter;
	vec2 wSize;
public:
	Camera2D() : wCenter(0, 0), wSize(2, 2) { }

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
	void Follow(vec2 t) {
		if (track)
			wCenter = t;
	}
};


Camera2D camera;

float tCurrent = 0;
GPUProgram gpuProgram;
const int nTesselatedVertices = 1000;

class Body {
	unsigned int vao;
	vec2 points[2];

public:

	Body(vec2 &p1, vec2 &p2) {
		points[0] = p1;
		points[1] = p2;
	}

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(points),
			points,
			GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);

		glVertexAttribPointer(0,
			2, GL_FLOAT,
			GL_FALSE,
			0, NULL);
	}

	vec2 getp1() { return points[0]; }
	vec2 getp2() { return points[1]; }

	void Move(float x, float y) {
		points[0].x = x;
		points[0].y = y;
	}

	void Draw() {


		mat4 MVPTransform(1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			points[0].x, points[0].y, 0.0f, 1.0f);


		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		if (colorLocation >= 0) glUniform3f(colorLocation, 1.0f, 0.0f, 0.0f);

		mat4 M = MVPTransform * camera.V()*camera.P();

		M.SetUniform(gpuProgram.getId(), "MVP");


		glBindVertexArray(vao);
		glDrawArrays(GL_LINE_STRIP, 0, 2);
	}
};




class Curve {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;

protected:
	std::vector<vec4> wCtrlPoints;
public:
	Curve() {

		glGenVertexArrays(1, &vaoCurve);
		glBindVertexArray(vaoCurve);

		glGenBuffers(1, &vboCurve);
		glBindBuffer(GL_ARRAY_BUFFER, vboCurve);

		glEnableVertexAttribArray(0);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);


		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);


	}

	virtual vec4 r(float t) { return wCtrlPoints[0]; }
	virtual vec4 rd(float t) { return wCtrlPoints[0]; }
	virtual vec4 re(float x) { return wCtrlPoints[0]; }
	virtual float tStart() { return 0; }
	virtual float tEnd() { return 1; }




	void swap(vec4 *v1, vec4 *v2) {
		vec4 temp = *v1;
		*v1 = *v2;
		*v2 = temp;
	}


	virtual void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints.push_back(wVertex);
		for (unsigned int i = 0; i < wCtrlPoints.size() - 0; i++) {
			int minimum = i;
			for (unsigned int j = i + 1; j < wCtrlPoints.size(); j++)
				if (wCtrlPoints[j].x < wCtrlPoints[minimum].x) {
					minimum = j;
				}
			swap(&wCtrlPoints[minimum], &wCtrlPoints[i]);
		}

	}



	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		VPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");



		if (wCtrlPoints.size() > 0) {
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 0, 0, 0);
			glPointSize(1.0f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}

		if (wCtrlPoints.size() >= 2) {
			std::vector<float> vertexData;
			float xt = -1.0f;
			for (int i = 0; i < nTesselatedVertices; i++) {
				float y = re(xt).y;
				vertexData.push_back(xt);
				vertexData.push_back(y);
				vertexData.push_back(xt);
				vertexData.push_back(-1.0f);

				xt += 0.002f;

			}

			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 0, 0, 0);
			glDrawArrays(GL_LINE_STRIP, 0, 2 * nTesselatedVertices);





		}
	}
};



class CatmullRom : public Curve {
	std::vector<float> ts;

	vec4 Hermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {

		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = (((p1 - p0) * 3)*(1.0 / ((t1 - t0)*(t1 - t0)))) - ((v1 + (v0 * 2))*(1.0 / (t1 - t0)));
		vec4 a3 = (((p0 - p1) * 2)*(1.0 / ((t1 - t0)*(t1 - t0)*(t1 - t0)))) + ((v1 + (v0))*(1.0 / ((t1 - t0)*(t1 - t0))));

		vec4 rt = (a3*((t - t0)*(t - t0)*(t - t0))) + (a2*((t - t0)*(t - t0))) + (a1*((t - t0))) + a0;



		return rt;
	}





	vec4 DHermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = (((p1 - p0) * 3)*(1.0 / ((t1 - t0)*(t1 - t0)))) - ((v1 + (v0 * 2))*(1.0 / (t1 - t0)));
		vec4 a3 = (((p0 - p1) * 2)*(1.0 / ((t1 - t0)*(t1 - t0)*(t1 - t0)))) + ((v1 + (v0))*(1.0 / ((t1 - t0)*(t1 - t0))));

		vec4 rt = (a3*(t - t0)*(t - t0) * 3) + (a2*(t - t0) * 2) + a1;
		return rt;
	}

public:
	void AddControlPoint(float cX, float cY) {
		ts.push_back((float)wCtrlPoints.size());
		Curve::AddControlPoint(cX, cY);
	}
	float tStart() { return ts[0]; }
	float tEnd() { return ts[wCtrlPoints.size() - 1]; }



	vec4 ve(int i) {
		vec4 tag1 = (wCtrlPoints[i + 1] - wCtrlPoints[i])*(1.0 / (wCtrlPoints[i + 1].x - wCtrlPoints[i].x));
		vec4 tag2 = (wCtrlPoints[i] - wCtrlPoints[i - 1])*(1.0 / (wCtrlPoints[i].x - wCtrlPoints[i - 1].x));
		vec4 vi = (tag1 + tag2)*(0.3f)*(-1);
		return vi;
	}

	vec4 re(float x) {
		vec4 v0;
		vec4 v1;

		for (int i = 0; i < wCtrlPoints.size() - 1; i++) {
			if (wCtrlPoints[i].x <= x && x <= wCtrlPoints[i + 1].x) {
				if (i == 0) {
					v0 = wCtrlPoints[1] - wCtrlPoints[0];
					v1 = ve(i + 1);

				}
				else {
					v0 = ve(i);
					if (i == wCtrlPoints.size() - 2)
						v1 = wCtrlPoints[i + 1] - wCtrlPoints[i];
					else v1 = ve(i + 1);

				}
				return Hermite(wCtrlPoints[i], v0, wCtrlPoints[i].x, wCtrlPoints[i + 1], v1, wCtrlPoints[i + 1].x, x);

			}

		}

		return vec4(0, 0, 0, 0);
	}



};


class Leg {
	unsigned int vao;
	unsigned int vbo;
	vec2 points[3];
	float l1;
	float l2;
	int dis = 1;

public:

	Leg(vec2 p1, vec2 p2, vec2 p3) {
		points[0] = p1;
		points[1] = p2;
		points[2] = p3;

		l1 = abs(length(p1) - length(p2));
		l2 = abs(length(p2) - length(p3));
	}

	void Move(float x, float y) {
		points[0].x = x;
		points[0].y = y;
	}

	void printl() {
		printf("\n*%.2f*\n", l1);
		printf("\n*%.2f*\n", l2);
	}

	void setdis() { dis *= -1; }

	void disort() {
		if (dis == 1) {
			points[0].x += 0.005f;
		}
		else if (dis == -1) {
			points[0].x -= 0.005f;
		}
	}

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);


		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(points),
			points,
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);

		glVertexAttribPointer(0,
			2, GL_FLOAT,
			GL_FALSE,
			0, NULL);
	}
	void Draw() {


		mat4 MVPTransform(1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			points[0].x, points[0].y, 0.0f, 1.0f);


		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		if (colorLocation >= 0) glUniform3f(colorLocation, 1.0f, 0.0f, 0.0f);

		mat4 M = MVPTransform * camera.V()*camera.P();

		M.SetUniform(gpuProgram.getId(), "MVP");



		glBindVertexArray(vao);
		glDrawArrays(GL_LINE_STRIP, 0, 3);
	}

};


class Head {
	unsigned int vao;
	unsigned int vbo;
	float radius = 0.05f;
	vec2 vertices[32 + 2];
	float center[2];
public:

	Head(float x, float y) {
		center[0] = x;
		center[1] = y;
		vertices[0] = vec2(center[0] + radius * cosf(float(0) / float(32)*float(2)*float(M_PI)), center[1] + radius * sinf(float(0) / float(32)*float(2)*float(M_PI)));
		for (int i = 1; i < 32 + 2; i++) {
			vertices[i] = vec2(center[0] + radius * cosf(float(i) / float(32)*float(2)*float(M_PI)), center[1] + radius * sinf(float(i) / float(32)*float(2)*float(M_PI)));
		}
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);


		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vertices),
			vertices,
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);

		glVertexAttribPointer(0,
			2, GL_FLOAT,
			GL_FALSE,
			0, NULL);
	}

	void Move(float x, float y) {
		center[0] = x;
		center[1] = y;
	}



	void Draw() {
		mat4 MVPTransform(1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			center[0], center[1], 0.0f, 1.0f);



		mat4 M = MVPTransform * camera.V()*camera.P();


		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		if (colorLocation >= 0) glUniform3f(colorLocation, 1.0f, 1.0f, 0.0f);

		M.SetUniform(gpuProgram.getId(), "MVP");
		glBindVertexArray(vao);
		glDrawArrays(GL_LINE_STRIP, 0, 32 + 1);



	}




};

class Circle {
	unsigned int vao;
	unsigned int vbo;
	int direction = 1;
	float radius;
	float force = 600;
	vec2 vertices[72 + 1];
	float center[2];
	vec2 translatepos;

	vec2 wTranslate;
	float phi = 0.0005;


public:

	Circle(float radius = float(0.1), float rX = 0.0, float rY = 0.0) {
		center[0] = rX;
		center[1] = rY;
		this->radius = radius;
		vertices[0] = vec2(center[0] + radius * cosf(float(0) / float(32)*float(2)*float(M_PI)), center[1] + radius * sinf(float(0) / float(32)*float(2)*float(M_PI)));
		for (int i = 1; i < 72 + 1; i++) {

			vertices[i] = vec2(center[0] + radius * cosf(float(i) / float(32)*float(2)*float(M_PI)), center[1] + radius * sinf(float(i) / float(32)*float(2)*float(M_PI)));
			if (i == 1 || i == 9 || i == 17 || i == 25) {

				vertices[i + 1] = vec2(center[0], center[1]);
				vertices[i + 2] = vec2(center[0] + radius * cosf(float(i) / float(32)*float(2)*float(M_PI)), center[1] + radius * sinf(float(i) / float(32)*float(2)*float(M_PI)));
				i++;
			}
			else {
				vertices[i + 1] = vec2(center[0] + radius * cosf(float(i) / float(32)*float(2)*float(M_PI)), center[1] + radius * sinf(float(i) / float(32)*float(2)*float(M_PI)));
				i++;
			}

		}

	}

	vec2 gettr() { return translatepos; }
	vec2 getv() { return vertices[3]; }
	float getphi() { return phi; }
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);



		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vertices),
			vertices,
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);

		glVertexAttribPointer(0,
			2, GL_FLOAT,
			GL_FALSE,
			0, NULL);



	}

	void Animate(Curve *c) {

		if (center[0] >= 0.9f) {
			direction *= -1;

		}
		if (center[0] <= -0.9f) {
			direction *= -1;

		}


		vec2 before = vec2(center[0] - 0.05f, c->re(center[0] - 0.05f).y);

		vec2 after = vec2(center[0] + 0.05f, c->re(center[0] + 0.1f).y);

		vec2 diff = after - before;
		float alpha = atan(diff.y / diff.x);
		float xg = diff.x * (-force + (50 * 9.81*sin(alpha)));

		phi = (xg / 100000) / radius;
		Rotate();

		if (direction == 1) {
			center[0] -= xg / 100000;

		}
		else if (direction == -1) {
			force *= -1;
			center[0] += xg / 100000;

		}


		diff = vec2(diff.x / sqrt(diff.x * diff.x + diff.y * diff.y), diff.y / sqrt(diff.x * diff.x + diff.y * diff.y));
		vec2 normal = vec2(diff.y, -diff.x);
		translatepos = vec2(center[0] - normal.x * radius, c->re(center[0]).y - normal.y * radius);


	}

	void Rotate() {
		for (int i = 0; i < 73; i++)
		{
			vec4 newPos = vec4(vertices[i].x, vertices[i].y, 0, 1)*mat4(cosf(phi), sinf(phi), 0, 0,
				-sinf(phi), cosf(phi), 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1);
			vertices[i].x = newPos.x;
			vertices[i].y = newPos.y;

			glBindBuffer(GL_ARRAY_BUFFER, vbo);

			glBufferData(GL_ARRAY_BUFFER,
				sizeof(vertices),
				vertices,
				GL_DYNAMIC_DRAW);

		}
	}

	void Move(float x, float y) {
		center[0] = x;
		center[1] = y;
	}

	float getr() { return length(vertices[32]); }

	void Draw() {
		mat4 MVPTransform(1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);

		mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		mat4 M = TranslateMatrix(vec3(translatepos.x, translatepos.y, 0.0f)) *camera.V()*camera.P();


		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		if (colorLocation >= 0) glUniform3f(colorLocation, 1.0f, 1.0f, 0.0f);

		M.SetUniform(gpuProgram.getId(), "MVP");
		glBindVertexArray(vao);
		glDrawArrays(GL_LINES, 0, 72 + 1);




	}

};


Curve * curve;
Circle c;
vec2 * pos = new vec2(1.0f, 5.0f);
vec2 p1 = vec2(0.0f, 0.35f);
vec2 p2 = vec2(0.0f, 0.1f);
vec4 N;
vec4 T;
Body * b = new Body(p1, p2);
Head * h;
Leg * l = new Leg(vec2(0.0f, 0.1f), vec2(0.1f, 0.05f), vec2(0.0f, 0.03f));
float xc = 0.0;
float yc = 0.0;



void onInitialization() {
	printf("%.2f", pos->x);
	printf("%.2f", pos->y);

	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	curve = new CatmullRom();
	curve->AddControlPoint(-1.0f, 0.75f);
	curve->AddControlPoint(0.0f, 0.0f);
	curve->AddControlPoint(1.0f, 0.5f);
	c.Create();
	b->Create();
	h = new Head(b->getp1().x, b->getp1().y + 0.05);
	h->Create();
	l->Create();
	l->printl();



	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0.1, 0.5, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	c.Draw();
	curve->Draw();
	b->Draw();
	h->Draw();
	l->Draw();
	glutSwapBuffers();
}


void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ')
		track = true;
	else if (key == 'd')
		camera.Pan(vec2(-0.05f, 0.0f));
	glutPostRedisplay();
}


void onKeyboardUp(unsigned char key, int pX, int pY) {

}

int pickedControlPoint = -1;

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		curve->AddControlPoint(cX, cY);
		glutPostRedisplay();
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		glutPostRedisplay();
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {

	}

	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		l->setdis();

		glutPostRedisplay();
	}
}


void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
}


void onIdle() {
	static float tend = 0;
	const float dt = 0.01;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	xc += 0.00001f;
	yc += 0.00001f;
	c.Animate(curve);
	b->Move(c.gettr().x, c.gettr().y);
	h->Move(b->getp1().x, b->getp1().y);
	l->Move(b->getp1().x, b->getp1().y);
	camera.Follow(b->getp1());
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
	}
	glutPostRedisplay();

	//objects.push_back(new Mirror(vec3(-1.0f, 0.0f,0.0f), vec3(1.0f,-1.0f,0.0f), rmaterial));
		//objects.push_back(new Mirror(vec3(1.0f, 0.0f, 0.0f), vec3(-1.0f, -1.0f, 0.0f), rmaterial));
		//objects.push_back(new Mirror(vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), rmaterial));
}