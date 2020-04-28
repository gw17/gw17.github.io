import processing.core.*;

public class RandomBoid {

	private PVector pos;
	private PVector velocity;
	private float maxSpeed;
	private PApplet graphicalContext;
	
	public RandomBoid(PApplet p, int x, int y) {
		this.pos = new PVector(x, y);
		int v1 = (int) (Math.random() * 2) - 1;
		int v2 = (int) (Math.random() * 2) - 1;
		this.velocity = new PVector(v1, v2);

		this.maxSpeed = 2;
		this.graphicalContext = p;
	}

	public void updatePosition() {
		PVector acc = this.randomForce();
		
		this.velocity.add(acc);
		this.velocity.limit(this.maxSpeed);
		this.pos.add(this.velocity);
	}

	public PVector randomForce() {
		int lower = -10;
		int upper = 10;
		
		int v1 = (int) (Math.random() * (upper - lower)) + lower;
		int v2 = (int) (Math.random() * (upper - lower)) + lower;
		
		return new PVector(v1, v2);
	}
	
	/**
	 * Fonction g�rant l'affichage
	 **/
	public void run() {
		this.updatePosition();
		this.borders();
		this.render();
	}

	public void render() {
		// Draw a triangle rotated in the direction of velocity
		float r = (float) 2.0;
		float theta = this.velocity.heading() + PConstants.PI / 2;
		graphicalContext.fill(200, 100);
		graphicalContext.stroke(255);
		graphicalContext.pushMatrix();
		graphicalContext.translate(this.pos.x, this.pos.y);
		graphicalContext.rotate(theta);
		graphicalContext.beginShape(PConstants.TRIANGLES);
		graphicalContext.vertex(0, -r * 2);
		graphicalContext.vertex(-r, r * 2);
		graphicalContext.vertex(r, r * 2);
		graphicalContext.endShape();
		graphicalContext.popMatrix();
	}

	public void borders() {
		float r = (float) 2.0;
		if (this.pos.x < -r) {
			this.pos.x = graphicalContext.width + r;
		}
		
		if (this.pos.y < -r) {
			this.pos.y = graphicalContext.height + r;
		}
		
		if (this.pos.x > graphicalContext.width + r) {
			this.pos.x = -r;
		}
		
		if (this.pos.y > graphicalContext.height + r) {
			this.pos.y = -r;
		}
	}

}
