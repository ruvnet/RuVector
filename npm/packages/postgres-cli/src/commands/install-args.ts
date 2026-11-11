/**
 * Validation and argv builders for install/server-management commands.
 *
 * Every value here comes from CLI flags and ends up in a child process, so
 * it is either checked against a strict charset or passed as a discrete
 * argv element / environment variable — never spliced into a shell string.
 */

export const SUPPORTED_PG_VERSIONS = ['14', '15', '16', '17'];

export function assertPgVersion(v: string): string {
  if (!SUPPORTED_PG_VERSIONS.includes(v)) {
    throw new Error(`Unsupported PostgreSQL version: ${v}. Use one of ${SUPPORTED_PG_VERSIONS.join(', ')}.`);
  }
  return v;
}

/** Docker container names: [a-zA-Z0-9][a-zA-Z0-9_.-]* */
export function assertContainerName(name: string): string {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/.test(name)) {
    throw new Error(`Invalid container name: ${name}`);
  }
  return name;
}

/** Docker image tags: [A-Za-z0-9_][A-Za-z0-9_.-]{0,127} */
export function assertImageTag(tag: string): string {
  if (!/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/.test(tag)) {
    throw new Error(`Invalid image version tag: ${tag}`);
  }
  return tag;
}

/** PostgreSQL user/database names accepted by the installer. */
export function assertPgIdentifier(name: string, what = 'identifier'): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]{0,62}$/.test(name)) {
    throw new Error(`Invalid ${what}: ${name}. Use letters, digits and underscores (max 63).`);
  }
  return name;
}

export function assertPort(port: number): number {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`Invalid port: ${port}`);
  }
  return port;
}

/** SQL string literal (standard_conforming_strings, the default since PG 9.1). */
export function sqlLiteral(s: string): string {
  return `'${s.replace(/'/g, "''")}'`;
}

export interface SudoStep {
  /** argv for `sudo` */
  args: string[];
  /** SQL fed on stdin (keeps secrets out of argv / `ps`) */
  input?: string;
}

/**
 * Steps for native configuration. Each psql invocation is its own session,
 * so the extension step connects to the target database with -d.
 */
export function configureSteps(user: string, password: string, database: string): SudoStep[] {
  assertPgIdentifier(user, 'user');
  assertPgIdentifier(database, 'database');
  const psql = ['-u', 'postgres', 'psql', '-X', '-q'];
  return [
    // Without ON_ERROR_STOP, an already-existing user does not block the database step.
    {
      args: psql,
      input:
        `CREATE USER "${user}" WITH PASSWORD ${sqlLiteral(password)} SUPERUSER;\n` +
        `CREATE DATABASE "${database}" OWNER "${user}";\n`,
    },
    {
      args: [...psql, '-v', 'ON_ERROR_STOP=1', '-d', database, '-c', 'CREATE EXTENSION IF NOT EXISTS ruvector;'],
    },
  ];
}

export interface DockerRunSpec {
  name: string;
  port: number;
  user: string;
  password: string;
  database: string;
  image: string;
  dataDir?: string;
}

/** `docker run` argv; the password travels via the child env, not argv. */
export function dockerRunArgs(spec: DockerRunSpec): { args: string[]; env: NodeJS.ProcessEnv } {
  const args = [
    'run', '-d', '--name', assertContainerName(spec.name),
    '-p', `${assertPort(spec.port)}:5432`,
    '-e', `POSTGRES_USER=${assertPgIdentifier(spec.user, 'user')}`,
    '-e', 'POSTGRES_PASSWORD',
    '-e', `POSTGRES_DB=${assertPgIdentifier(spec.database, 'database')}`,
  ];
  if (spec.dataDir) {
    // A ':' past the drive prefix would be parsed by -v as a mount target/option.
    if (spec.dataDir.replace(/^[A-Za-z]:[\\/]/, '').includes(':')) {
      throw new Error(`Invalid data directory (contains ':'): ${spec.dataDir}`);
    }
    args.push('-v', `${spec.dataDir}:/var/lib/postgresql/data`);
  }
  args.push(spec.image);
  return { args, env: { ...process.env, POSTGRES_PASSWORD: spec.password } };
}

/** `docker exec <name> psql ... -c <sql>` with the SQL as one argv element. */
export function dockerPsqlArgs(name: string, sql: string): string[] {
  return ['exec', assertContainerName(name), 'psql', '-U', 'ruvector', '-d', 'ruvector', '-c', sql];
}
