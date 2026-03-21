/**
 * DrAgnes Deployment Runbook
 *
 * Structured deployment procedures, cost model, monitoring configuration,
 * and rollback strategies for the DrAgnes classification service.
 */

/** Deployment step definition */
export interface DeploymentStep {
	name: string;
	command: string;
	timeout: string;
	description: string;
	rollbackCommand?: string;
	requiresApproval?: boolean;
}

/** Rollback procedure */
export interface RollbackProcedure {
	trigger: string;
	steps: DeploymentStep[];
	maxRollbackTimeMinutes: number;
}

/** Monitoring endpoint */
export interface MonitoringEndpoint {
	name: string;
	url: string;
	interval: string;
	alertThreshold: string;
}

/** Per-practice cost breakdown at different scale tiers */
export interface PracticeScaleCost {
	/** Cost per practice at 10 practices */
	at10: number;
	/** Cost per practice at 100 practices */
	at100: number;
	/** Cost per practice at 1000 practices */
	at1000: number;
}

/** Monthly infrastructure cost breakdown */
export interface InfraBreakdown {
	cloudRun: number;
	firestore: number;
	gcs: number;
	pubsub: number;
	cdn: number;
	scheduler: number;
	monitoring: number;
}

/** Revenue tier pricing */
export interface RevenueTier {
	starter: number;
	professional: number;
	enterprise: string;
	academic: number;
	underserved: number;
}

/** Cost model for DrAgnes deployment */
export interface CostModel {
	/** Per-practice cost at various scales (USD/month) */
	perPractice: PracticeScaleCost;
	/** Monthly infrastructure breakdown (USD) */
	breakdown: InfraBreakdown;
	/** Monthly subscription revenue tiers (USD) */
	revenue: RevenueTier;
	/** Number of practices needed to break even */
	breakEven: number;
}

/** Complete deployment runbook */
export interface DeploymentRunbook {
	prerequisites: string[];
	steps: DeploymentStep[];
	rollback: RollbackProcedure;
	secrets: string[];
	monitoring: {
		endpoints: MonitoringEndpoint[];
		dashboardUrl: string;
		oncallChannel: string;
	};
	costModel: CostModel;
}

/**
 * DrAgnes production deployment runbook.
 *
 * Covers build, containerization, deployment to Cloud Run,
 * health checks, smoke tests, rollback, and cost modeling.
 */
export const DEPLOYMENT_RUNBOOK: DeploymentRunbook = {
	prerequisites: [
		"Node.js >= 20.x installed",
		"Docker >= 24.x installed",
		"gcloud CLI authenticated with ruv-dev project",
		"Access to gcr.io/ruv-dev container registry",
		"All secrets configured in Google Secret Manager",
		"CI pipeline green on main branch",
		"Changelog updated with version notes",
		"ADR-117 compliance checklist completed",
	],

	steps: [
		{
			name: "Build",
			command: "npm run build",
			timeout: "5m",
			description: "Build the SvelteKit application with DrAgnes modules",
		},
		{
			name: "Run Tests",
			command: "npm test -- --run",
			timeout: "3m",
			description: "Execute full test suite including DrAgnes classifier and benchmark tests",
		},
		{
			name: "Docker Build",
			command:
				"docker build -f Dockerfile.dragnes -t gcr.io/ruv-dev/dragnes:$VERSION .",
			timeout: "10m",
			description: "Build production Docker image with WASM CNN module",
			rollbackCommand: "docker rmi gcr.io/ruv-dev/dragnes:$VERSION",
		},
		{
			name: "Push Image",
			command: "docker push gcr.io/ruv-dev/dragnes:$VERSION",
			timeout: "5m",
			description: "Push container image to Google Container Registry",
		},
		{
			name: "Deploy to Staging",
			command: [
				"gcloud run deploy dragnes-staging",
				"--image gcr.io/ruv-dev/dragnes:$VERSION",
				"--region us-central1",
				"--memory 2Gi",
				"--cpu 2",
				"--min-instances 0",
				"--max-instances 10",
				"--set-secrets OPENROUTER_API_KEY=openrouter-key:latest,OPENAI_BASE_URL=openai-base-url:latest",
				"--allow-unauthenticated",
			].join(" "),
			timeout: "3m",
			description: "Deploy to staging Cloud Run service for validation",
			rollbackCommand:
				"gcloud run services update-traffic dragnes-staging --to-revisions LATEST=0",
		},
		{
			name: "Staging Health Check",
			command: "curl -f https://dragnes-staging.ruv.io/health",
			timeout: "30s",
			description: "Verify staging service is responsive and healthy",
		},
		{
			name: "Staging Smoke Test",
			command: [
				"curl -sf -X POST https://dragnes-staging.ruv.io/api/v1/analyze",
				'-H "Content-Type: application/json"',
				'-d \'{"image":"data:image/png;base64,iVBOR...","magnification":10}\'',
			].join(" "),
			timeout: "30s",
			description: "Run classification on a test image against staging",
		},
		{
			name: "Deploy to Production",
			command: [
				"gcloud run deploy dragnes",
				"--image gcr.io/ruv-dev/dragnes:$VERSION",
				"--region us-central1",
				"--memory 2Gi",
				"--cpu 2",
				"--min-instances 1",
				"--max-instances 50",
				"--set-secrets OPENROUTER_API_KEY=openrouter-key:latest,OPENAI_BASE_URL=openai-base-url:latest",
				"--allow-unauthenticated",
			].join(" "),
			timeout: "3m",
			description: "Deploy to production Cloud Run service",
			requiresApproval: true,
			rollbackCommand:
				"gcloud run services update-traffic dragnes --to-revisions LATEST=0",
		},
		{
			name: "Production Health Check",
			command: "curl -f https://dragnes.ruv.io/health",
			timeout: "30s",
			description: "Verify production service health endpoint",
		},
		{
			name: "Production Smoke Test",
			command: [
				"curl -sf -X POST https://dragnes.ruv.io/api/v1/analyze",
				'-H "Content-Type: application/json"',
				'-d \'{"image":"data:image/png;base64,iVBOR...","magnification":10}\'',
			].join(" "),
			timeout: "30s",
			description: "Run classification on a test image against production",
		},
	],

	rollback: {
		trigger:
			"Health check failure, error rate > 5%, latency p99 > 10s, or classification accuracy drop > 10%",
		steps: [
			{
				name: "Revert Traffic",
				command:
					"gcloud run services update-traffic dragnes --to-revisions PREVIOUS=100",
				timeout: "1m",
				description: "Route 100% traffic back to the previous stable revision",
			},
			{
				name: "Verify Rollback",
				command: "curl -f https://dragnes.ruv.io/health",
				timeout: "30s",
				description: "Confirm the previous revision is healthy",
			},
			{
				name: "Notify On-Call",
				command:
					'curl -X POST $SLACK_WEBHOOK -d \'{"text":"DrAgnes rollback triggered for $VERSION"}\'',
				timeout: "10s",
				description: "Alert the on-call team about the rollback",
			},
		],
		maxRollbackTimeMinutes: 5,
	},

	secrets: [
		"OPENROUTER_API_KEY",
		"OPENAI_BASE_URL",
		"MCP_SERVERS",
		"MONGODB_URL",
		"SESSION_SECRET",
		"WEBHOOK_SECRET",
	],

	monitoring: {
		endpoints: [
			{
				name: "Health",
				url: "https://dragnes.ruv.io/health",
				interval: "30s",
				alertThreshold: "2 consecutive failures",
			},
			{
				name: "Classification Latency",
				url: "https://dragnes.ruv.io/metrics/latency",
				interval: "1m",
				alertThreshold: "p99 > 5000ms",
			},
			{
				name: "Error Rate",
				url: "https://dragnes.ruv.io/metrics/errors",
				interval: "1m",
				alertThreshold: "> 5% of requests",
			},
			{
				name: "Model Accuracy",
				url: "https://dragnes.ruv.io/metrics/accuracy",
				interval: "1h",
				alertThreshold: "< 75% on validation set",
			},
		],
		dashboardUrl: "https://console.cloud.google.com/monitoring/dashboards/dragnes",
		oncallChannel: "#dragnes-oncall",
	},

	costModel: {
		perPractice: {
			at10: 25.80,
			at100: 7.52,
			at1000: 3.89,
		},
		breakdown: {
			cloudRun: 130,
			firestore: 50,
			gcs: 15,
			pubsub: 5,
			cdn: 20,
			scheduler: 1,
			monitoring: 10,
		},
		revenue: {
			starter: 99,
			professional: 199,
			enterprise: "custom",
			academic: 0,
			underserved: 0,
		},
		breakEven: 30,
	},
};

/**
 * Calculate total monthly infrastructure cost.
 */
export function calculateMonthlyCost(model: CostModel): number {
	const b = model.breakdown;
	return b.cloudRun + b.firestore + b.gcs + b.pubsub + b.cdn + b.scheduler + b.monitoring;
}

/**
 * Calculate monthly revenue at a given number of practices.
 *
 * Assumes a mix: 60% starter, 30% professional, 10% enterprise (at $499).
 */
export function calculateMonthlyRevenue(
	practiceCount: number,
	model: CostModel
): number {
	const starterCount = Math.floor(practiceCount * 0.6);
	const proCount = Math.floor(practiceCount * 0.3);
	const enterpriseCount = practiceCount - starterCount - proCount;

	return (
		starterCount * model.revenue.starter +
		proCount * model.revenue.professional +
		enterpriseCount * 499
	);
}
