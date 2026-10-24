import path from "node:path";
import { promises as fs } from "node:fs";

export type DichotomyPreference = "E" | "I" | "S" | "N" | "T" | "F" | "J" | "P";

export interface AssessmentAnswer {
  questionNumber: number;
  choice: string;
}

export interface DichotomyResult {
  preference: DichotomyPreference;
  ci: number;
  ciCategory: string;
  theta: number;
  dichotomyName: string;
}

export interface ScoringPayload {
  dichotomyResults: Record<string, DichotomyResult>;
}

export async function calculateAssessmentResults(
  answers: Record<number, AssessmentAnswer>,
  questionSet: unknown
): Promise<ScoringPayload> {
  const scorer = await import("./scorer.js");
  return scorer.calculateResults(answers, questionSet) as ScoringPayload;
}

export async function loadFormMQuestions() {
  const filePath = path.join(process.cwd(), "src", "modules", "psychometrics", "questions.json");
  const content = await fs.readFile(filePath, "utf8");
  return JSON.parse(content);
}
